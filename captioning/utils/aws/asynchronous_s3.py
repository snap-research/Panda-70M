# From https://gist.github.com/fherbine/0d4aa473e5cc2c5f6f8a0e1b35a62625

# See https://github.com/boto/boto3/issues/3113
from concurrent.futures import ThreadPoolExecutor

import os
import queue
import sys
import threading
import time

import boto3


class AsynchronousS3:
    """Download/Upload asynchronously files from/to AWS S3 bucket.
    Example:
    >>> from asynchronous_s3 import AsynchronousS3
    >>>
    >>> def my_success_callback(size, duration):
    ...     print(f'My {size} bytes file has been uploaded in {duration} sec.')
    ...
    >>> def upload_to_s3():
    ...     async_s3 = AsynchronousS3('my-bucket-name')
    ...     async_s3.upload_file(
    ...         'path/to_my/local_file',
    ...         'object/key',
    ...         on_success=my_success_callback,
    ...         on_failure=lambda error: print(error),
    ...     )
    ...     print('code to be executed...')
    ...
    >>> upload_file()
    code to be executed...
    My 105673 bytes file has been uploaded in 5.3242523 sec.
    >>>
    """
    def __init__(self, s3_name, max_concurrent_processes: int, *args, **kwargs):
        """Class constructor.
        arguments:
        s3_name -- Name of the bucket. Please check your credentials before
        (~/.aws/credentials)
        max_concurrent_processes -- The maximum number of concurrent uploads and downloads
        args -- extra arguments to give to boto3.session.Session instance.
        Keywords arguments:
        kwargs -- extra kwargs to give to boto3.session.Session instance.
        """
        session = boto3.session.Session(*args, **kwargs)
        service_resource = session.resource('s3')
        # Alternative way to get the resource
        #service_resource = boto3.resource('s3')
        self.bucket = service_resource.Bucket(s3_name)
        self.max_concurrent_processes = max_concurrent_processes
        self._io_threads_queue = threads_queue = queue.Queue()
        self.execution_info_lock = threading.BoundedSemaphore(1)
        self.internal_id = 0 # Unique ids to assign to threads
        # Map from thread ids to time of start
        self.execution_info = {
        }
        self._daemon = _S3Daemon(threads_queue, self.execution_info_lock, self.execution_info)
        self._daemon.start()

    def add_execution_info_thread(self, thread):
        """
        Registers the thread for execution, waiting until room for thread is made
        """
        
        thread_id = thread.internal_id

        success = False
        while not success:
            self.execution_info_lock.acquire()
            # There is space for execution
            if len(self.execution_info) < self.max_concurrent_processes:
                self.execution_info[thread_id] = time.time()
                success = True
            self.execution_info_lock.release()
            # Wait before retry
            if not success:
                time.sleep(.3)

    def wait_for_termination(self):
        """
        Waits for the termination of all threads
        """

        success = False
        while not success:
            self.execution_info_lock.acquire()
            # Waits until the daemon removes all execution info.
            # This happens either because of timeouts or because of threads finish their job
            if len(self.execution_info) == 0:
                success = True
            self.execution_info_lock.release()
            # Wait before retry
            if not success:
                time.sleep(.3)

    def upload_file(self, local_path, key, on_start=None, on_success=None, on_failure=None,
                    **kwargs):
        """Upload a file from your computer to s3.
        Arguments:
        local_path -- Source path on your computer.
        key -- AWS S3 destination object key. More info:
        https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingMetadata.html
        Keywords arguments:
        on_success -- success callback to call. Given arguments will be:
        file_size and duration. Default is `None`, any callback is called.
        on_failure -- failure callback to call. Given arguments will be:
        error_message. Default is `None`, any callback is called.
        kwargs -- Extra kwargs for standard boto3 Bucket `upload_file` method.
        """
        bucket = self.bucket
        method = bucket.upload_file

        self.internal_id += 1

        thread = _S3Thread(
            self.internal_id,
            method,
            on_start=on_start,
            on_success=on_success,
            on_failure=on_failure,
            threads_queue=self._io_threads_queue,
            Key=key,
            Filename=local_path,
            **kwargs,
        )
        # Records that the thread started and waits until there is room for execution
        self.add_execution_info_thread(thread)
        thread.start()

    def dowload_file(self, local_path, key, on_start=None, on_success=None, on_failure=None,
                     **kwargs):
        """Download a file from S3 to your computer.
        Arguments:
        local_path -- Destination path on your computer.
        key -- AWS S3 source object key. More info:
        https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingMetadata.html
        Keywords arguments:
        on_success -- success callback to call. Given arguments will be:
        file_size and duration. Default is `None`, any callback is called.
        on_failure -- failure callback to call. Given arguments will be:
        error_message. Default is `None`, any callback is called.
        kwargs -- Extra kwargs for standard boto3 Bucket `download_file` method
        """
        bucket = self.bucket
        method = bucket.download_file

        self.internal_id += 1
        thread = _S3Thread(
            self.internal_id,
            method,
            on_start=on_start,
            on_success=on_success,
            on_failure=on_failure,
            threads_queue=self._io_threads_queue,
            Key=key,
            Filename=local_path,
            **kwargs,
        )
        # Records that the thread started and waits until there is room for execution
        self.add_execution_info_thread(thread)
        thread.start()

    def exit(self):
        self._daemon.exit()

    def __del__(self):
        self.exit()


class _S3Thread(threading.Thread):
    def __init__(self, internal_id, method, on_start, on_success, on_failure, threads_queue,
                 *args, **kwargs):
        self.internal_id = internal_id
        self._method = method
        self._on_start = on_start
        self._on_success = on_success
        self._on_failure = on_failure
        self._threads_queue = threads_queue
        self._meth_args = args
        self._meth_kwargs = kwargs

        self._start_time = time.time()

        super().__init__()

    def run(self):
        method = self._method
        args = self._meth_args
        kwargs = self._meth_kwargs

        try:
            self._on_start()
            #print("- debug: pretending to call async S3 method with args: {}, kwargs: {}".format(args, kwargs), flush=True)
            method(*args, **kwargs)
            self.success()
        except Exception as error:
            self.failed(error)

    def success(self):
        file_path = self._meth_kwargs['Filename']
        file_size = os.path.getsize(file_path)
        duration = time.time() - self._start_time

        self.stop(self._on_success, file_size, duration)

    def failed(self, error_message):
        self.stop(self._on_failure, error_message)

    def stop(self, callback, *args):
        
        try:
            if callback is not None:
                callback(*args)
        finally:
            # Kills self
            self._threads_queue.put(self)


class _S3Daemon(threading.Thread):
    def __init__(self, threads_queue, execution_info_lock, execution_info, max_thread_wait_seconds=10 * 60):
        self._threads_queue = threads_queue
        self.execution_info_lock = execution_info_lock
        self.execution_info = execution_info
        self.max_thread_wait_seconds = max_thread_wait_seconds

        self._running_event = threading.Event()
        self._running_event.set()

        super().__init__(daemon=True)

    def run(self):
        while self._running_event.is_set():
            time.sleep(.1)

            # Removes every thread that has hit the timeout assuming they are dead
            self.execution_info_lock.acquire()
            for thread_id, thread_time in self.execution_info.items():
                if time.time() - thread_time > self.max_thread_wait_seconds:
                    del self.execution_info[thread_id]
                    print("Warning: thread {} has been killed after {} seconds due to unresponsiveness".format(thread_id, self.max_thread_wait_seconds), file=sys.stderr)
            self.execution_info_lock.release()

            try:
                thread = self._threads_queue.get_nowait()
            except queue.Empty:
                continue

            thread.join(5)
            # Logs a warning if the thread is still alive
            if thread.is_alive():
                thread_id = thread.internal_id
                print("Warning: thread {} was joined but is still alive".format(thread_id), file=sys.stderr)

            # Removes the thread from the execution info
            self.execution_info_lock.acquire()
            thread_id = thread.internal_id
            if thread_id in self.execution_info:
                del self.execution_info[thread_id]
            else:
                print("Warning: thread {} has been killed but was already removed from execution".format(thread_id), file=sys.stderr)
            self.execution_info_lock.release()


    def exit(self):
        self._running_event.clear()
        self.join(5)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Debugging pyaws.s3')
    parser.add_argument(
        '--mode',
        '-m',
        required=False,
        help='dowload from S3. Default is `down` (download). It could be `up`',
        type=str,
        default='down',
    )
    parser.add_argument(
        '--bucket-name',
        '-b',
        required=True,
        type=str,
        help='AWS S3 bucket name.'
    )
    parser.add_argument(
        '--file',
        '-f',
        required=True,
        type=str,
        help='Local file dst/src.'
    )
    parser.add_argument(
        '--object-key',
        '-k',
        required=True,
        type=str,
        help='S3 Object key'
    )

    args = parser.parse_args()

    mode = 'download' if args.mode.lower() != 'up' else 'upload'
    bucket_name = args.bucket_name
    file_path = args.file
    object_key = args.object_key

    def on_success(file_size, duration):
        print(f'A {file_size} bytes has been {mode} in {duration} secs.')

    def on_failure(error):
        print('An error occured: %s' % error)

    s3 = AsynchronousS3(bucket_name)

    if mode == 'download':
        s3.dowload_file(
            local_path=file_path,
            key=object_key,
            on_success=on_success,
            on_failure=on_failure,
        )
    else:
        s3.upload_file(
            local_path=file_path,
            key=object_key,
            on_success=on_success,
            on_failure=on_failure,
        )