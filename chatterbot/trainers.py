import os
import sys
import csv
import glob
import json
import tarfile
from tqdm import tqdm
from dateutil import parser as date_parser
from chatterbot.conversation import Statement


class Trainer(object):
    """
    Base class for all other trainer classes.

    :param boolean show_training_progress: Show progress indicators for the
           trainer. The environment variable ``CHATTERBOT_SHOW_TRAINING_PROGRESS``
           can also be set to control this. ``show_training_progress`` will override
           the environment variable if it is set.
    """

    def __init__(self, chatbot, **kwargs):
        self.chatbot = chatbot

        environment_default = bool(int(os.environ.get('CHATTERBOT_SHOW_TRAINING_PROGRESS', True)))

        self.disable_progress = not kwargs.get(
            'show_training_progress',
            environment_default
        )

    def get_preprocessed_statement(self, input_statement):
        """
        Preprocess the input statement.
        """
        for preprocessor in self.chatbot.preprocessors:
            input_statement = preprocessor(input_statement)

        return input_statement

    def train(self, *args, **kwargs):
        """
        This method must be overridden by a child class.
        """
        raise self.TrainerInitializationException()

    class TrainerInitializationException(Exception):
        """
        Exception raised when a base class has not overridden
        the required methods on the Trainer base class.
        """

        def __init__(self, message=None):
            default = (
                'A training class must be specified before calling train(). '
                'See https://docs.chatterbot.us/training/'
            )
            super().__init__(message or default)

    def _generate_export_data(self):
        result = []
        for statement in self.chatbot.storage.filter():
            if statement.in_response_to:
                result.append([statement.in_response_to, statement.text])

        return result

    def export_for_training(self, file_path='./export.json'):
        """
        Create a file from the database that can be used to
        train other chat bots.
        """
        export = {'conversations': self._generate_export_data()}
        with open(file_path, 'w+', encoding='utf8') as jsonfile:
            json.dump(export, jsonfile, ensure_ascii=False)


class ListTrainer(Trainer):
    """
    Allows a chat bot to be trained using a list of strings
    where the list represents a conversation.
    """

    def train(self, conversation):
        """
        Train the chat bot based on the provided list of
        statements that represents a single conversation.
        """
        previous_statement_text = None
        previous_statement_search_text = ''

        statements_to_create = []

        # Run the pipeline in bulk to improve performance
        documents = self.chatbot.tagger.as_nlp_pipeline(conversation)

        # for text in enumerate(conversation):
        for document in tqdm(documents, desc='List Trainer', disable=self.disable_progress):
            statement_search_text = document._.search_index

            statement = self.get_preprocessed_statement(
                Statement(
                    text=document.text,
                    search_text=statement_search_text,
                    in_response_to=previous_statement_text,
                    search_in_response_to=previous_statement_search_text,
                    conversation='training'
                )
            )

            previous_statement_text = statement.text
            previous_statement_search_text = statement_search_text

            statements_to_create.append(statement)

        self.chatbot.storage.create_many(statements_to_create)


class ChatterBotCorpusTrainer(Trainer):
    """
    Allows the chat bot to be trained using data from the
    ChatterBot dialog corpus.
    """

    def train(self, *corpus_paths):
        from chatterbot.corpus import load_corpus, list_corpus_files

        data_file_paths = []

        # Get the paths to each file the bot will be trained with
        for corpus_path in corpus_paths:
            data_file_paths.extend(list_corpus_files(corpus_path))

        for corpus, categories, _file_path in tqdm(
            load_corpus(*data_file_paths),
            desc='ChatterBot Corpus Trainer',
            disable=self.disable_progress
        ):
            statements_to_create = []

            # Train the chat bot with each statement and response pair
            for conversation in corpus:

                # Run the pipeline in bulk to improve performance
                documents = self.chatbot.tagger.as_nlp_pipeline(conversation)

                previous_statement_text = None
                previous_statement_search_text = ''

                for document in documents:
                    statement_search_text = document._.search_index

                    statement = Statement(
                        text=document.text,
                        search_text=statement_search_text,
                        in_response_to=previous_statement_text,
                        search_in_response_to=previous_statement_search_text,
                        conversation='training'
                    )

                    statement.add_tags(*categories)

                    statement = self.get_preprocessed_statement(statement)

                    previous_statement_text = statement.text
                    previous_statement_search_text = statement_search_text

                    statements_to_create.append(statement)

            if statements_to_create:
                self.chatbot.storage.create_many(statements_to_create)


class GenericFileTrainer(Trainer):
    """
    Allows the chat bot to be trained using data from a CSV or JSON file,
    or directory of those file types.
    """

    def __init__(self, chatbot, **kwargs):
        """
        data_path: str The path to the data file or directory.
        field_map: dict A dictionary containing the column name to header mapping.
        """
        super().__init__(chatbot, **kwargs)

        # File path or directory
        self.data_path = kwargs.get('data_path')

        self.file_extension = None

        # NOTE: If the key is an integer, this be the
        # column index instead of the key or header
        DEFAULT_STATEMENT_TO_HEADER_MAPPING = {
            'text': 'text',
            'conversation': 'conversation',
            'created_at': 'created_at',
            'persona': 'persona',
            'tags': 'tags'
        }

        self.field_map = kwargs.get(
            'field_map',
            DEFAULT_STATEMENT_TO_HEADER_MAPPING
        )

    def _get_file_list(self, limit):
        """
        Get a list of files to read from the data set.
        """

        if self.file_extension is None:
            raise self.TrainerInitializationException(
                'The file_extension attribute must be set before calling train().'
            ) 

        # List all csv or json files in the specified directory
        if os.path.isdir(self.data_path):
            glob_path = os.path.join(self.data_path, '**', f'*.{self.file_extension}')

            # TODO: Use iglob instead of glob for better performance with large directories
            data_files = glob.glob(glob_path, recursive=True)
        else:
            data_files = [self.data_path]

        if limit is not None:
            data_files = data_files[:limit]

        return data_files

    def train(self, limit=None):
        """
        Train a chatbot with data from the data file.

        limit: int If defined, the maximum number of files to read from the data set.
        """

        if self.data_path is None:
            raise self.TrainerInitializationException(
                'The data_path argument must be set to the path of a file or directory.'
            )

        data_files = self._get_file_list(limit)

        if not data_files:
            self.chatbot.logger.warning(
                'No [{}] files were detected at: {}'.format(
                    self.file_extension,
                    self.data_path
                )
            )

        for data_file in tqdm(data_files, desc='Training', disable=self.disable_progress):

            previous_statement_text = None
            previous_statement_search_text = ''

            file_extension = data_file.split('.')[-1].lower()

            statements_to_create = []

            with open(data_file, 'r', encoding='utf-8') as file:

                if self.file_extension == 'json':
                    data = json.load(file)
                    data = data['conversation']
                elif file_extension == 'csv':
                    use_header = bool(isinstance(next(iter(self.field_map.values())), str))

                    if use_header:
                        data = csv.DictReader(file)
                    else:
                        data = csv.reader(file)
                elif file_extension == 'tsv':
                    use_header = bool(isinstance(next(iter(self.field_map.values())), str))

                    if use_header:
                        data = csv.DictReader(file, delimiter='\t')
                    else:
                        data = csv.reader(file, delimiter='\t')
                else:
                    self.logger.warning(f'Skipping unsupported file type: {file_extension}')
                    continue

                text_row = self.field_map['text']

                documents = self.chatbot.tagger.as_nlp_pipeline([
                    (
                        row[text_row],
                        {
                            # Include any defined metadata columns
                            key: row[value]
                            for key, value in self.field_map.items()
                            if key != text_row
                        }
                    ) for row in data if len(row) > 0
                ])

            for document, context in documents:
                statement = Statement(
                    text=document.text,
                    conversation=context.get('conversation', 'training'),
                    persona=context.get('persona', None),
                    tags=context.get('tags', [])
                )

                if 'created_at' in context:
                    statement.created_at = date_parser.parse(context['created_at'])

                statement.search_text = document._.search_index
                statement.search_in_response_to = previous_statement_search_text

                # Use the in_response_to attribute for the previous statement if
                # one is defined, otherwise use the last statement which was created
                if 'in_response_to' in self.field_map.keys():
                    statement.in_response_to = context.get(self.field_map['in_response_to'], None)
                else:
                    statement.in_response_to = previous_statement_text

                for preprocessor in self.chatbot.preprocessors:
                    statement = preprocessor(statement)

                previous_statement_text = statement.text
                previous_statement_search_text = statement.search_text

                statements_to_create.append(statement)

            self.chatbot.storage.create_many(statements_to_create)


class CsvFileTrainer(GenericFileTrainer):
    """
    Allow chatbots to be trained with data from a CSV file or
    directory of CSV files.

    TSV files are also supported, as long as the file_extension
    parameter is set to 'tsv'.
    """

    def __init__(self, chatbot, **kwargs):
        super().__init__(chatbot, **kwargs)

        self.file_extension = kwargs.get('file_extension', 'csv')


class JsonFileTrainer(GenericFileTrainer):
    """
    Allow chatbots to be trained with data from a JSON file or
    directory of JSON files.
    """

    def __init__(self, chatbot, **kwargs):
        super().__init__(chatbot, **kwargs)

        self.file_extension = 'json'

        DEFAULT_STATEMENT_TO_KEY_MAPPING = {
            'text': 'text',
            'conversation': 'conversation',
            'created_at': 'created_at',
            'in_response_to': 'in_response_to',
            'persona': 'persona',
            'tags': 'tags'
        }

        self.field_map = kwargs.get(
            'field_map',
            DEFAULT_STATEMENT_TO_KEY_MAPPING
        )


class UbuntuCorpusTrainer(CsvFileTrainer):
    """
    Allow chatbots to be trained with the data from the Ubuntu Dialog Corpus.

    For more information about the Ubuntu Dialog Corpus visit:
    https://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/
    """

    def __init__(self, chatbot, **kwargs):
        super().__init__(chatbot, **kwargs)
        home_directory = os.path.expanduser('~')

        self.data_download_url = kwargs.get(
            'ubuntu_corpus_data_download_url',
            'http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz'
        )

        self.data_directory = kwargs.get(
            'ubuntu_corpus_data_directory',
            os.path.join(home_directory, 'ubuntu_data')
        )

        # Directory containing extracted data
        self.data_path = os.path.join(
            self.data_directory, 'ubuntu_dialogs'
        )

        self.field_map = {
            'text': 3,
            'created_at': 0,
            'persona': 1,
        }

    def is_downloaded(self, file_path):
        """
        Check if the data file is already downloaded.
        """
        if os.path.exists(file_path):
            self.chatbot.logger.info('File is already downloaded')
            return True

        return False

    def is_extracted(self, file_path):
        """
        Check if the data file is already extracted.
        """

        if os.path.isdir(file_path):
            self.chatbot.logger.info('File is already extracted')
            return True
        return False

    def download(self, url, show_status=True):
        """
        Download a file from the given url.
        Show a progress indicator for the download status.
        """
        import requests

        # Create the data directory if it does not already exist
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        file_name = url.split('/')[-1]
        file_path = os.path.join(self.data_directory, file_name)

        # Do not download the data if it already exists
        if self.is_downloaded(file_path):
            return file_path

        with open(file_path, 'wb') as open_file:
            print('Downloading %s' % url)
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:
                # No content length header
                open_file.write(response.content)
            else:
                for data in tqdm(
                    response.iter_content(chunk_size=4096),
                    desc='Downloading',
                    disable=not show_status
                ):
                    open_file.write(data)

        print('Download location: %s' % file_path)
        return file_path

    def extract(self, file_path):
        """
        Extract a tar file at the specified file path.
        """
        print('Extracting {}'.format(file_path))

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        def track_progress(members):
            sys.stdout.write('.')
            for member in members:
                # This will be the current file being extracted
                yield member

        with tarfile.open(file_path) as tar:
            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar, path=self.data_path, members=track_progress(tar))

        self.chatbot.logger.info('File extracted to {}'.format(self.data_path))

        return True
    
    def _get_file_list(self, limit):
        """
        Get a list of files to read from the data set.
        """
        # Download and extract the Ubuntu dialog corpus if needed
        corpus_download_path = self.download(self.data_download_url)

        # Extract if the directory does not already exist
        if not self.is_extracted(self.data_path):
            self.extract(corpus_download_path)

        extracted_corpus_path = os.path.join(
            self.data_path, '**', '**', '*.tsv'
        )

        if limit is not None:
            return glob.glob(extracted_corpus_path)[:limit]

        return glob.glob(extracted_corpus_path)
