import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.modeling_llama import LlamaForCausalLM
# from video_llama.models.Qformer import BertEncoder
from transformers import LlamaTokenizer, BertConfig
from transformers import StoppingCriteria, StoppingCriteriaList
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
from video_llama.models.ImageBind.models.imagebind_model import ImageBindModel, ModalityType
from video_llama.models.ImageBind.models import imagebind_model
# from flamingo_pytorch import PerceiverResampler

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
@registry.register_model("video_llama")
class VideoLLAMA(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_text_Qformer(cls, num_query_token, text_width, num_hidden_layers=2, input_vid2tex_query_embed=True):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.num_attention_heads = 16
        encoder_config.encoder_width = text_width
        encoder_config.hidden_size = text_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)

        if input_vid2tex_query_embed:
            return Qformer
        else:
            query_tokens = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            return Qformer, query_tokens

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        input_prompt=False,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        max_caption_len=48,
        max_prompt_len=100,
        start_sym="<s>",
        end_sym="</s>",
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

        frozen_llama_proj=True,
        frozen_video_Qformer=True,
        frozen_text_Qformer=True,
        
        llama_proj_model="",
        fusion_header_type="seqTransf",
        max_frame_pos=32,
        fusion_head_layers=2,
        num_video_query_token=32,
        num_text_query_token=32,
        input_vid2tex_query_embed=True,
        detach_video_query_embed=True,
        imagebind_ckpt_path=""
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.input_prompt = input_prompt

        logging.info('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        logging.info('Loading VIT Done')

        logging.info('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.llama_tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16, # torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16, # torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')


        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)
            logging.info('Loaded LLAMA proj from pretrained checkpoint')
        else:
            logging.info('Did NOT load LLAMA proj from pretrained checkpoint')

        if frozen_llama_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading LLAMA proj Done')

        self.max_caption_len = max_caption_len
        self.max_prompt_len = max_prompt_len
        self.start_sym = start_sym
        self.end_sym = end_sym

        logging.info('Initializing Video Qformer')
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)
        self.num_video_query_token = num_video_query_token
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=self.Qformer.config.hidden_size, num_hidden_layers=2)

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if frozen_video_Qformer:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            logging.info('Video Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('Video Qformer is not frozen')
        logging.info('Initializing Video Qformer Done')

        if self.input_prompt:
            logging.info('Initializing Text Qformer')
            self.text_prompt_position_embedding = nn.Embedding(max_prompt_len, self.llama_model.config.hidden_size)
            self.num_text_query_token = num_text_query_token
            self.input_vid2tex_query_embed = input_vid2tex_query_embed
            self.detach_video_query_embed = detach_video_query_embed
            if self.input_vid2tex_query_embed:
                assert num_video_query_token == num_text_query_token
                self.text_Qformer = self.init_text_Qformer(
                    num_query_token=self.num_text_query_token, \
                    text_width=self.llama_model.config.hidden_size, \
                    num_hidden_layers=2, \
                    input_vid2tex_query_embed=self.input_vid2tex_query_embed
                )
            else:
                self.text_Qformer, self.text_query_tokens = self.init_text_Qformer(
                    num_query_token=self.num_text_query_token, \
                    text_width=self.llama_model.config.hidden_size, \
                    num_hidden_layers=2, \
                    input_vid2tex_query_embed=self.input_vid2tex_query_embed
                )

            self.text_Qformer.cls = None
            self.text_Qformer.bert.embeddings.word_embeddings = None
            self.text_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.text_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

            if frozen_text_Qformer:
                for name, param in self.text_Qformer.named_parameters():
                    param.requires_grad = False
                for name, param in self.text_prompt_position_embedding.named_parameters():
                    param.requires_grad = False
                if not self.input_vid2tex_query_embed:
                    self.text_query_tokens.requires_grad = False
                logging.info('Text Qformer is frozen')
            else:
                for name, param in self.text_Qformer.named_parameters():
                    param.requires_grad = True
                for name, param in self.text_prompt_position_embedding.named_parameters():
                    param.requires_grad = True
                if not self.input_vid2tex_query_embed:
                    self.text_query_tokens.requires_grad = True
                logging.info('Text Qformer is not frozen')
            logging.info('Initializing Text Qformer Done')
        else:
            logging.info('NOT initializing Text Qformer')
            
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_videoQformer_visual(self, image):
        device = image.device

        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # frame attention
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama
    
    def encode_textQformer_prompt(self, prompts, vid_embeds):
        self.llama_tokenizer.pad_token = self.llama_tokenizer.bos_token
        self.llama_tokenizer.truncation_side = "left"
        self.llama_tokenizer.padding_side = "left"
        device = vid_embeds.device
        prompts = [self.start_sym + t for t in prompts]
        prompt_tokens = self.llama_tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_prompt_len,
            add_special_tokens=False
        ).to(device)
        prompt_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids)
        prompt_atts = prompt_tokens.attention_mask

        batch_size, num_tokens, _ = prompt_embeds.shape
        position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.text_prompt_position_embedding(position_ids)
        prompt_embeds = position_embeds + prompt_embeds

        if self.input_vid2tex_query_embed:
            if not self.detach_video_query_embed:
                query_embeds = vid_embeds.type(prompt_embeds.dtype)
            else:
                query_embeds = vid_embeds.detach().type(prompt_embeds.dtype)
        else:
            query_embeds = self.text_query_tokens.expand(batch_size, -1, -1)

        text_query_output = self.text_Qformer.bert(
            query_embeds=query_embeds,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_atts,
            return_dict=True,
            )
        prompt_embeds = text_query_output.last_hidden_state
        prompt_atts = torch.ones(prompt_embeds.size()[:-1], dtype=torch.long).to(device)

        return prompt_embeds, prompt_atts

    def forward(self, samples, return_pred=False):
        if 'conv_type' in samples.keys() and samples['conv_type']=='multi':
            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            if len(image.size())==4:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)

            num_patch_tokens = self.num_video_query_token
            img_embeds, atts_img = self.encode_videoQformer_visual(image)
                
            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

            new_input_embeds=[]
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]

                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patch_tokens, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                
                cur_image_idx+=1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            return {"loss": loss}
        else:
            output_dict = {}

            # embed video
            image = samples["image"]
            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)
            img_embeds, img_atts = self.encode_videoQformer_visual(image)

            # embed <bos>
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                              dtype=torch.long).to(image.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            bos_atts = img_atts[:, :1]

            # embed prompt
            if self.input_prompt:
                prompt = samples["prompt"]
                txt_embeds, txt_atts = self.encode_textQformer_prompt(prompt, img_embeds)
                bos_atts = txt_atts[:, :1]
                mixed_embeds = torch.cat([bos_embeds, txt_embeds, img_embeds], dim=1)
                mixed_atts = torch.cat([bos_atts, txt_atts, img_atts], dim=1)
            else:
                mixed_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
                mixed_atts = torch.cat([bos_atts, img_atts], dim=1)            

            # embed caption
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.truncation_side = "right"
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["caption"]]
            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_caption_len,
                add_special_tokens=False
            ).to(image.device)
            to_regress_tokens.input_ids[:,-1].fill_(self.llama_tokenizer.eos_token_id) # force the last token to be <eos>
            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

            # concat all the embedding
            inputs_embeds = torch.cat([mixed_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([mixed_atts, to_regress_tokens.attention_mask], dim=1)
            
            # make target
            targets = to_regress_tokens.input_ids.masked_fill(
                ~to_regress_tokens.attention_mask.bool(), -100
            )
            empty_targets = (
                torch.ones([mixed_atts.shape[0], mixed_atts.shape[1]],
                        dtype=torch.long).to(image.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            output_dict["loss"] = outputs.loss

            if return_pred:
                stop_words_ids = [torch.tensor([2]).to(image.device)]
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

                output_dict["pred"] = []
                output_dict["target"] = samples["caption"]
                output_dict["video_path"] = samples["video_path"]

                outputs = self.llama_model.generate(
                    inputs_embeds=mixed_embeds,
                    max_new_tokens=300,
                    stopping_criteria=stopping_criteria,
                    num_beams=3,
                    do_sample=True,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    temperature=1.5,
                )
                
                for output_token in outputs:
                    if output_token[0] == self.llama_tokenizer.bos_token_id:  # remove bos token <s> at the beginning.
                        output_token = output_token[1:]
                    output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
                    output_text = output_text.split(self.end_sym)[0].strip()
                    output_dict["pred"].append(output_text)

        return output_dict

    def inference(self, image, prompt=None):
        # embed video
        if len(image.size()) != 5:
            time = 1
            image = einops.repeat(image, 'b c h w -> b c t h w',t = time)
        img_embeds, img_atts = self.encode_videoQformer_visual(image)

        # embed <bos>
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                            dtype=torch.long).to(image.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        bos_atts = img_atts[:, :1]

        # embed prompt
        if self.input_prompt:
            txt_embeds, txt_atts = self.encode_textQformer_prompt(prompt, img_embeds)
            bos_atts = txt_atts[:, :1]
            mixed_embeds = torch.cat([bos_embeds, txt_embeds, img_embeds], dim=1)
            mixed_atts = torch.cat([bos_atts, txt_atts, img_atts], dim=1)
        else:
            mixed_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
            mixed_atts = torch.cat([bos_atts, img_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).to(image.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        outputs = self.llama_model.generate(
            inputs_embeds=mixed_embeds,
            max_new_tokens=300,
            stopping_criteria=stopping_criteria,
            num_beams=3,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.5,
        )
        
        prediction = []
        for output_token in outputs:
            if output_token[0] == self.llama_tokenizer.bos_token_id:  # remove bos token <s> at the beginning.
                output_token = output_token[1:]
            output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split(self.end_sym)[0].strip()
            prediction.append(output_text)

        return prediction

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        input_prompt = cfg.get("input_prompt", False)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        max_caption_len = cfg.get("max_caption_len", 48)
        max_prompt_len = cfg.get("max_prompt_len", 100)
        start_sym = cfg.get("start_sym", "<s>")
        end_sym = cfg.get("end_sym", "</s>")
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_text_Qformer = cfg.get("frozen_text_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", "")

        fusion_header_type = cfg.get("fusion_header_type", "seqTransf")
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token = cfg.get("num_video_query_token", 32)
        num_text_query_token = cfg.get("num_text_query_token", 32)
        input_vid2tex_query_embed = cfg.get("input_vid2tex_query_embed", True)
        detach_video_query_embed = cfg.get("detach_video_query_embed", True)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", "/mnt/workspace/ckpt")
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            input_prompt=input_prompt,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            max_caption_len=max_caption_len,
            max_prompt_len=max_prompt_len,
            start_sym=start_sym,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            llama_proj_model=llama_proj_model,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_text_Qformer=frozen_text_Qformer,
            num_video_query_token=num_video_query_token,
            num_text_query_token=num_text_query_token,
            input_vid2tex_query_embed=input_vid2tex_query_embed,
            detach_video_query_embed=detach_video_query_embed,
            imagebind_ckpt_path=imagebind_ckpt_path,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path_2 = cfg.get("ckpt_2", "")  
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model
