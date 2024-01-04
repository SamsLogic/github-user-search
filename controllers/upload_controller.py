import os
from common.utils import send_response
from flask import request
from werkzeug.utils import secure_filename
from config.log import upload_cnt_logger as logger, CustomAdapterUpload
from scripts.github_reader import githubReader
from threading import Thread

def upload_controller():
    try:
        if (
            "file" not in request.files
            or request.files["file"].filename == ''
            or os.path.splitext(request.files["file"].filename)[1] != '.xlsx'
        ):
            return send_response(
                {
                    "message": "file is invalid or not provided"
                },
                400,
            )
        if (
            "project" not in request.form
            or type(request.form["project"]) != str
            or len(request.form["project"]) == 0
        ):
            return send_response(
                {
                    "message": "project is invalid"
                },
                400,
            )
        if (
            "old" in request.form
            and (type(request.form["old"]) != str
            or request.form["old"] not in ['true', 'false'])
        ):
            return send_response(
                {
                    "message": "old is invalid"
                },
                400,
            )
        
        f = request.files['file']
        filename = secure_filename(f.filename)
        file_path =  f'data/{filename}'
        f.save(file_path)
        project = request.form["project"]
        if "old" in request.form:
            old = request.form["old"] == 'true'
        else:
            old = False
        
        upload_logger = CustomAdapterUpload(logger, {'project': project})
        
        upload_logger.info("Staring upload")
        thread = Thread(
            target=githubReader(
                file_path=file_path,
                project=project,
                old=old,
            ).get_user_github_data
        )
        thread.start()
        # output = githubReader(file_path=file_path, project=project, old=old).get_user_github_data()
        # upload_logger.info("Finished upload")        

        # if output is None:
        #     upload_logger.exception("Something seems to have failed, output is None")
        #     return send_response(
        #         {
        #             "message": "Something seems to have failed"
        #         },
        #         500,
        #     )
        # else:
        upload_logger.info("Successfully uploaded")
        return send_response(
            {
                "message": "User parsing started"
            },
            200,
        )
    except Exception as e:
        upload_logger.exception(f'Failed with error: {str(e)}')
        return send_response({
            "message": f'{str(e)}'
        }, 500)

def status_controller():
    try:
        if (
            "project" not in request.json
            or type(request.json["project"]) != str
            or len(request.json["project"]) == 0
        ):
            return send_response(
                {
                    "message": "project is invalid"
                },
                400,
            )
        
        project = request.json["project"]

        with open(f'status_{project}.txt', 'r') as f:
            status = f.read()
        return send_response(
            {
                "message": "Status fetched",
                "status": status
            },
            200,
        )
    except Exception as e:
        return send_response({
            "message": f'{str(e)}'
        }, 500)