import os
import requests
import pandas as pd
from typing import Tuple
from config.config import GITHUB_API_KEY, LANGUAGE_MAP
from config.log import upload_logger as logger, CustomAdapterUpload
from scripts.preprocessor import preprocessor
from scripts.embeddings import embedder
from scripts.old_embeddings import embedder as old_embedder

class githubReader():
    def __init__(self, file_path: str, project: str, save: bool=True, data_path: str = None, old: bool = False) -> None:
        self.file_path = file_path
        self.project = project
        self.save = save
        self.language_map = LANGUAGE_MAP
        self.user_data = None
        self.preprocessor = preprocessor(project=project)
        if old:
            self.embedder = old_embedder(project=project)
        else:
            self.embedder = embedder(project=project)
        self.embed_col = 'data'
        self.upload_logger = CustomAdapterUpload(logger, {'project': project})
        self.upload_logger.info('Reading data from excel')
        if data_path is not None:
            self.user_data = pd.read_excel(data_path)
        else:
            self.__extract_users()

    def __extract_users(self) -> None:
        print(self.file_path)
        self.users = pd.read_excel(self.file_path)['names']

    def __get_code_and_markdown_files(self, username: str) -> Tuple[list, list]:
        files = []
        code_data = []

        headers = {
                'Authorization': f'Bearer {GITHUB_API_KEY}',
                'Content-Type': "application/vnd.github+json"
            }
        self.upload_logger.info(f'Getting data for {username}')
        repos_url = f'https://api.github.com/users/{username}/repos'
        response = requests.get(
            repos_url,
            headers = headers
            )
        response.raise_for_status()

        self.upload_logger.info(f'Getting repos for {username}')
        for repo in response.json():
            repo_name = repo['name']
            if 'languages' in repo and repo['langauges'] is not None:
              lang_resp = requests.get(repo['languages_url'], headers = {'Authorization': f'Bearer {GITHUB_API_KEY}', 'Content-Type': "application/vnd.github+json"})
              lang = ', '.join(list(lang_resp.json().keys()))
            else:
              lang = ''
            stars = repo['stargazers_count']
            files_url = f'https://api.github.com/repos/{username}/{repo_name}/contents'
            self.upload_logger.debug(f'Getting files for {repo_name}')
            while True:

                files_response = requests.get(
                      files_url,
                      headers = headers
                    )

                if files_response.status_code not in [200, 404]:
                  files_response.raise_for_status()
                elif files_response.status_code == 404 or files_response.status_code == 500:
                  break

                files_data = files_response.json()

                for file_info in files_data:
                    file_name = file_info['name']
                    if file_name.endswith(('.md')):
                        self.upload_logger.debug(f'Getting data for {file_name}')
                        if file_info['download_url'] is not None:
                            text = requests.get(file_info['download_url'], headers=headers).text
                        else:
                            text = ''
                        code_data.append('')
                        files.append({'repo': repo_name, 'filename': file_name, 'download_url': file_info['download_url'], 'size':file_info['size'], 'user':username, 'text_data':text, 'languages':lang, 'stars':stars})
                    elif file_name.endswith(('.py', '.js', '.java')):
                        self.upload_logger.debug(f'Getting data for {file_name}')
                        if file_info['download_url'] is not None:
                            text = requests.get(file_info['download_url'], headers=headers).text
                        else:
                            text = ''
                        code_data.append(text)
                        files.append({'repo': repo_name, 'filename': file_name, 'download_url': file_info['download_url'], 'size':file_info['size'], 'user':username, 'text_data':'', 'languages':lang, 'stars':stars})
                    else:
                        self.upload_logger.debug(f'File {file_name} is not a code or markdown file')
                        continue

                if 'next' in files_response.links:
                    files_url = files_response.links['next']['url']
                else:
                    break

        return files, code_data
    
    def __process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[~data.groupby(['user_repo'])['code_data'].transform("sum").isin(['', 0])]
        data['languages'] = data['languages'].str.lower()
        data['code_data'] = data['code_data'].fillna('')
        data['text_data'] = data['text_data'].fillna('')
        data[self.embed_col] = data['text_data'] + data['code_data']
        data['code_data'] = data['code_data'].str.replace('\\\\+n', '\\n', regex=True)
        data['file_lang'] = data['filename'].apply(lambda x: self.language_map[x.split('.')[-1]] if x.split('.')[-1] in self.language_map else '')
        return data

    def update_status(self, status):
        with open(f"status_{self.project}.txt", "w") as f:
            f.write(status)

    def get_user_github_data(self) -> pd.DataFrame:

        if os.path.exists(f'data/{self.project}_data.xlsx'):
            self.user_data = pd.read_excel(f'data/{self.project}_data.xlsx')
            self.user_data = self.user_data.fillna('')
        else:
            self.upload_logger.info('Getting data for all users')
            if self.user_data is None:
                user_data = []
                code_data = []
                self.update_status(f"0 out {len(self.users)} completed")
                for ind, username in enumerate(self.users, start=1):
                    self.upload_logger.debug(f'Getting data for {username}')
                    files, code_data_temp = self.__get_code_and_markdown_files(username)
                    user_data.extend(files)
                    code_data.extend(code_data_temp)
                    self.update_status(f"{ind} out {len(self.users)} completed")
                    
                user_data = pd.DataFrame(user_data)
                user_data['code_data'] = code_data
                self.user_data = user_data.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
            self.upload_logger.info('Processing data')
            self.user_data['user_repo'] = self.user_data['user'] + '_' + self.user_data['repo']
            self.user_data = self.__process_data(self.user_data)

            self.upload_logger.info('Preprocessing data')
            self.user_data = self.preprocessor.perprocess(self.user_data)
            if self.save:
                self.upload_logger.info('Saving data')
                self.user_data.to_excel(f'data/{self.project}_data.xlsx')

        self.user_data['strings'] = self.user_data['text_strings'] + self.user_data['code_strings']
        self.upload_logger.info('Embedding data')
        self.update_status(f"Embedding data")
        self.embedder.embed(self.user_data, self.embed_col)
        self.update_status(f"Embedding calculated")
        self.upload_logger.info('Embedding calculated')
        return True

if __name__ == '__main__':
    reader = githubReader(data_path='/home/user/Documents/Personal/projectMERC/data_wc_100_new.xlsx', save=False)
    user_data = reader.get_user_github_data()
    user_data.to_excel('/home/user/Documents/Personal/projectMERC/data_wc_100_new_read.xlsx', index=False)