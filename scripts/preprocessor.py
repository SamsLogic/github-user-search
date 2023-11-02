import pandas as pd
import re
import code_tokenize
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from tree_sitter import Language, Parser
from config.log import preprocessor_logger as logger, CustomAdapterUpload

# Load the Python grammar for Tree-sitter
Language.build_library(
  # Store the library in the `./my-languages.so` file
  'build/my-languages.so',

  # Include one or more languages
  [
    'build/tree-sitter-python',
    'build/tree-sitter-javascript',
    'build/tree-sitter-java'
  ]
)

# Specify the language (Python) for parsing
PY_LANGUAGE = Language('build/my-languages.so', 'python')
JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')

class parseCode():
    def __init__(self, project):
        self.langmap = {'python':PY_LANGUAGE, 'javascript':JS_LANGUAGE, 'java':JAVA_LANGUAGE}
        self.project = project
        self.parse_code_logger = CustomAdapterUpload(logger.getChild('parseCode'), {'project': project})
        
    def parse_code(self, source_code, language):
        parser = Parser()
        parser.set_language(self.langmap[language])

        # Parse the source code
        tree = parser.parse(bytes(source_code, 'utf8'))

        return tree

    def get_chunks(self, node):
        if node.type == 'class_definition' or node.type == 'function_definition' or node.type == 'for_statement' or node.type == 'call_expression':
            yield node

        for child in node.children:
            yield from self.get_chunks(child)

    def extract_chunks(self, source_code, language):
        tree = self.parse_code(source_code, language)

        # Get the root node of the AST
        root_node = tree.root_node

        # Find all class, function, and for loop nodes
        chunks = list(self.get_chunks(root_node))

        # Extract the chunk code
        out = []
        for chunk_node in chunks:
            start = chunk_node.start_byte
            end = chunk_node.end_byte
            if len(source_code[start:end].split(' ')) < 7:
                continue
            out.append(source_code[start:end])
        return out

class codeProcessor():
    def __init__(self, text_df, project, save = False):
        self.text_df = text_df
        self.text_df['code_data'] = self.text_df['code_data'].fillna('')
        self.function_extractor = parseCode(project=project)
        self.save = save
        self.project = project
        self.code_logger = CustomAdapterUpload(logger.getChild('codeProcessor'), {'project': project})

    def __get_code_block(self):
        new_df = {
            'user':[],
            'repo':[],
            'filename':[],
            'code':[],
        }
        for text_ind, text in enumerate(self.text_df['code_data']):
            if text == '':
                new_df['user'].append(self.text_df.iloc[text_ind]['user'])
                new_df['repo'].append(self.text_df.iloc[text_ind]['repo'])
                new_df['filename'].append(self.text_df.iloc[text_ind]['filename'])
                new_df['code'].append('')
                continue
            else:
                chunks = self.function_extractor.extract_chunks(text, self.text_df.iloc[text_ind]['file_lang'])
                for chunk in chunks:
                    new_df['user'].append(self.text_df.iloc[text_ind]['user'])
                    new_df['repo'].append(self.text_df.iloc[text_ind]['repo'])
                    new_df['filename'].append(self.text_df.iloc[text_ind]['filename'])
                    new_df['code'].append(chunk)
            if text_ind % 300 == 0:
                self.code_logger.info(f'Extracted code blocks for {text_ind} of {len(self.text_df)}')
        return pd.DataFrame(new_df)

    def __split_string(self, input_string):
        # Split underscored string
        if '_' in input_string:
            return ' '.join(input_string.split('_'))
        elif '-' in input_string:
            return ' '.join(input_string.split('-'))
        # Split camel cased string
        else:
            return ' '.join(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', input_string))
    
    def __chunk_function(self, text, language):
        tokenized_function = code_tokenize.tokenize(text, language, syntax_error=False)
        return tokenized_function

    def __extract_token_string(self, text, language):
        token_string = []
        if text == '':
            return ''
        tokenized_function = self.__chunk_function(text, language)
        for token in tokenized_function:
            if token.type in ['identifier']:
                token_string.append(self.__split_string(str(token)))
        return ' '.join(token_string)
    
    def process_df(self):
        self.code_logger.info('Getting code blocks from source code')
        text_df_func = self.__get_code_block()
        self.text_df = self.text_df.merge(text_df_func, on=['user', 'repo', 'filename'], how='left')
        self.text_df['code'] = self.text_df['code'].fillna('')

        self.code_logger.info('Extracting token strings from code blocks')
        self.text_df['code_strings'] = self.text_df.apply(lambda x: self.__extract_token_string(x['code'], x['file_lang']), axis=1)
        
        if self.save:
            self.code_logger.info('Saving code data')
            self.text_df.to_excel(f'data/{self.project}_data_code.xlsx', index=False)
        return self.text_df

class textProcessor():
    def __init__(self, data, project, save = False, save_path = ''):
        self.lemmatizer = WordNetLemmatizer()
        self.text_df = data
        self.text_df['text_data'] = self.text_df['text_data'].fillna('')
        self.project = project
        self.save = save
        self.save_path = save_path
        self.text_logger = CustomAdapterUpload(logger.getChild('textProcessor'), {'project': project})
    
    def __split_string(self, input_string):
        # Split underscored string
        if '_' in input_string:
            return input_string.split('_')
        # Split camel cased string
        else:
            return re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', input_string)

    def __noun_phrase_extraction(self, text):
        blob = [word for (word, pos) in pos_tag(word_tokenize(text)) if pos[0] == 'N']
        updated_blob = []
        for token in blob:
            updated_blob.extend(self.__split_string(str(token)))
        updated_blob = [self.lemmatizer.lemmatize(word) for word in updated_blob if len(word) > 1]
        updated_blob = ' '.join(updated_blob)
        if updated_blob == '':
            return ''
        return updated_blob
    
    def process_df(self):
        token_strings = []
        self.text_logger.info('Extracting noun phrases from text data')
        for _, text in enumerate(self.text_df['text_data']):
            if text == '':
                token_strings.append('')
                continue
            token_strings.append(self.__noun_phrase_extraction(text))
        self.text_df['text_strings'] = token_strings
        if self.save:
            self.text_logger.info('Saving text data')
            self.text_df.to_excel(f'data/{self.project}_data_text.xlsx', index=False)
        return self.text_df

    def process_text(self, text):
        self.text_logger.info('Extracting noun phrases from text data')
        return self.__noun_phrase_extraction(text)

class preprocessor():
    def __init__(self, project):
        self.project = project
        self.preprocess_logger = CustomAdapterUpload(logger.getChild('preprocessor'), {'project': project})
    
    def perprocess(self, data):
        self.preprocess_logger.info('Preprocessing text data')
        text_parser = textProcessor(data, self.project)
        data = text_parser.process_df()

        self.preprocess_logger.info('Preprocessing code data')
        code_parser = codeProcessor(data, self.project)
        data = code_parser.process_df()
        return data
