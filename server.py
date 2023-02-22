from sanic import Sanic, response
import subprocess
import app as App

App.init()

server = Sanic("my_app")


@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})

# Inference POST handler at '/' is called for every http call from Banana


@server.route('/', methods=["POST"])
def inference(request):

    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json

    output = App.training(model_inputs)

    return response.json(output)


if __name__ == '__main__':
    server.run(host='0.0.0.0', port="8000", workers=1)
