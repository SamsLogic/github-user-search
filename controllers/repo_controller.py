from common.utils import send_response
from flask import request
from config.log import repo_cnt_logger as logger, CustomAdapter
from scripts.repos import getRepos

def repo_controller():
    try:
        if (
            "query" not in request.json
            or type(request.json["query"]) != str
            or len(request.json["query"]) == 0
        ):
            return send_response(
                {
                    "message": "query is invalid",
                    "data": [],
                    "cost": 0
                },
                400,
            )
        if (
            "project" not in request.json
            or type(request.json["project"]) != str
            or len(request.json["project"]) == 0
        ):
            return send_response(
                {
                    "message": "project is invalid",
                    "data": [],
                    "cost": 0
                },
                400,
            )
        if (
            "top_n" in request.json
            and (type(request.json["top_n"]) != int
            or request.json["top_n"] <= 0)
        ):
            return send_response(
                {
                    "message": "top_n is invalid",
                    "data": []
                },
                400,
            )
        if (
            "min_stars" in request.json
            and (type(request.json["min_stars"]) != int
            or request.json["min_stars"] < 0)
        ):
            return send_response(
                {
                    "message": "min_stars is invalid",
                    "data": []
                },
                400,
            )
        if (
            "execution_id" in request.json
            and (type(request.json["execution_id"]) != int
            or request.json["execution_id"] < 0)
        ):
            return send_response(
                {
                    "message": "execution_id is invalid",
                    "data": []
                },
                400,
            )

        query = request.json["query"]
        project = request.json["project"]
        if "execution_id" in request.json:
            execution_id = request.json["execution_id"]
        else:
            execution_id = 1
        if "top_n" in request.json:
            top_n = request.json["top_n"]
        else:
            top_n = 5
        
        if "min_stars" in request.json:
            stars = request.json["min_stars"]
        else:
            stars = 0
        
        repo_logger = CustomAdapter(logger, {'executionId': execution_id, 'project': project})
        
        repo_logger.info("Staring repo search")
        output = getRepos(query=query, project=project, execution_id=execution_id, top_n=top_n, stars=stars).get_repos()
        repo_logger.info("Finished repo search")

        if output is None:
            repo_logger.exception("Something seems to have failed, output is None")
            return send_response(
                {
                    "message": "Something seems to have failed",
                    "data": []
                },
                500,
            )
        else:
            repo_logger.info("Successfull repo search")
            return send_response(
                {
                    "message": "Success",
                    "data": output
                },
                200,
            )
    except Exception as e:
        repo_logger.exception(f'Failed with error: {str(e)}')
        return send_response({
            "message": f'{str(e)}',
            "data": []
        }, 500)