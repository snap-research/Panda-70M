import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
import time


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)


    videos = ['makeup.mp4', 'barbie.mp4', 'table-tennis.mp4', 'cooking-seasoning.mp4', 'excel.mp4', 'racing-game.mp4', 'cooking-talking.mp4', 'shot-put.mp4', 'throw-football.mp4']
    '''
    texts = [['In the video, a young woman is shown applying makeup on her face while holding a compact.',
              'A young woman with long brown hair is seen applying makeup to her face with a brush, wearing a wedding dress.',
              'a woman is applying makeup to her face with a brush.',
              'A woman is putting on makeup and brushing her hair in front of a mirror.'],
             ['The video shows a young girl in a magical forest, with a dog, a magical dragon, and a magical tree in a magical forest.',
              'The main content of the video is an animated young girl, in various poses and locations, with pink backgrounds and a pink couch.',
              'a young girl in pink pajamas is sitting on a bed in a pink room.',
              'A girl in pink pajamas is reading a book while sitting on a small bed in a pink room, with a doll sitting on a couch nearby.'],
             ['In the video, young men play ping pong in an indoor arena with an umpire present.',
              'A tennis match takes place on a purple court with two tennis players playing a game of table tennis.',
              'people playing a game of tennis on a court with the ball in the air.',
              'A man and woman are playing tennis on a court, and they are also playing a video game with a ball on the same court.'],
             ['The video shows a woman cooking chicken in a pan with various ingredients, such as flour, water, and chicken.',
              'Topic: woman hand holding spoon in a bowl of black powder or spices on a black counter.',
              'a counter with small bowls of food and small plastic containers of sauce.',
              'A person is mixing a dish of potatoes and other ingredients in a frying pan, and a counter is shown with small bowls of food and plastic containers of sauce.'],
             ['The video shows a computer screen with a red background displaying the status of a job running on a computer.',
              'The video shows a window of an application with different screens and a white background with different icons and information displayed on it, such as software names, files and folders.',
              'screenshot of a screen with a number of options.',
              'The video shows multiple screenshots of a website with text, numbers, and options.'],
             ['The video is a 3D racing game with a person riding a motorcycle.',
              'A dirt bike is shown in a video game and then driving down a dirt road on a motorcycle.',
              'a man riding a motorcycle in a video game.',
              'The video shows screenshots and a man riding a motorcycle in a video game that is available for free on the app store and operating system software.'],
             ['The video shows two men dressed in chef uniforms standing behind a counter holding pies and talking with each other, with a person standing in front of them holding a knife.',
              'The video shows a group of young men having a party and making different dishes in the kitchen.',
              'a man holding a bowl of food next to another man in a kitchen.',
              'The video shows men in a kitchen holding bowls of food, dough, a glass bowl, and a loaf of bread while preparing food with a chef.'],
             ['A group of young people participating in an organized sporting event in a park.',
              'A crowd of people is gathered at a school or sporting event, including a young man wearing a blue shirt standing on a track.',
              'a crowd of people are playing frisbee on a field.',
              'The video shows a crowd of people playing frisbee on a field, as well as people playing baseball, flying kites, and playing basketball in a park.'],
             ['The video shows a man standing in a field in front of a pickup truck, looking at a soccer ball.',
              'A young man wearing a white shirt is playing soccer in the field in front of a truck.',
              'a man holding a frisbee in a field of grass.',
              'The video shows multiple people holding frisbees and playing with them in a grassy field, as well as a man holding a baseball bat and a kite flying in the sky.']]
    '''
    texts = [['A woman using a makeup brush to apply blush to her face.',
              'A woman using a brush to apply makeup on her face.',
              'A woman using a brush to apply makeup to her face.'],
             ['A girl laying on a couch reading a book.',
              'A cartoon girl sitting on a couch reading a book.',
              'A cartoon girl laying on a couch in a pink room.'],
             ['Two men playing ping pong on a tennis court.',
              'A couple of people playing ping pong on a court.',
              'Two men playing ping pong on a purple court.'],
             ['The video shows a person cooking with various ingredients such as pepper powder, soy sauce, and sesame oil.',
              'A person is preparing a meal of chicken with white pepper, green peppers, and soy sauce.',
              'The video shows a person preparing a dish by cutting up food in a metal bowl and adding white pepper powder.'],
             ['A screen shot of a computer screen.',
              'A screen shot of a computer screen.',
              'A screen shot of a computer screen.'],
             ['The video showcases a person playing a video game involving riding a dirt bike on a dirt road.',
              'The video depicts a person riding a motorcycle on a dirt road and a person riding a dirt bike in a video game.',
              'The video depicts a person riding a dirt bike in a video game using a controller.'],
             ['Two men are standing together in a kitchen.',
              'Two men are standing together in a kitchen.',
              'Two men are preparing food in a kitchen.'],
             ['A group of people are participating in various activities including walking, standing in a field, playing a game of frisbee, and watching a baseball game.',
              'The video depicts a group of people enjoying a day of frisbee together',
              'A group of people are gathered together to watch a man throw a frisbee.'],
             ['A blurry picture of a man throwing a baseball.',
              'A blurry picture of a man throwing a baseball.',
              'A blurry image of a person throwing a baseball at a truck.']]


    time_counter = 0
    for video, text in zip(videos, texts):
        start_time = time.time()

        # Load data
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text, device),
            ModalityType.VISION: data.load_and_transform_video_data(['../test_videos/%s'%video], device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        print(video, torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1))

        t = time.time() - start_time
        time_counter += t
        
    print('Average time: %.3f seconds'%(time_counter/len(videos)))
