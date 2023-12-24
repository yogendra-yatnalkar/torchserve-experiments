from ts.torch_handler.base_handler import BaseHandler
import torch, torchvision
import os
from torchvision import transforms
import torch.nn.functional as F
from torch.profiler import ProfilerActivity
import io
import base64
from PIL import Image

try:
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
except ImportError as error:
    XLA_AVAILABLE = False


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.context = None
        self.model_pt_path = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0
        self.preprocess = None
        self.profiler_args = {}

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        if context is not None and hasattr(context, "model_yaml_config"):
            self.model_yaml_config = context.model_yaml_config

        properties = context.system_properties
        print("Properties: ", properties)
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )
        elif XLA_AVAILABLE:
            self.device = xm.xla_device()
        else:
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        print("Manifest: ", self.manifest)
        print("Model Directory: ", model_dir)

        # loading the model
        model_wts_name = "vit_l_16.pt"
        model_weights_path = os.path.join(model_dir, model_wts_name)

        self.model = torchvision.models.vit_l_16()
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.to(self.device)
        self.model.eval()
        print("Model Loaded Successfully....")

        self.model_preprocess = (
            torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()
        )
        self.model_preprocess.antialias = True

    def data_preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """

        images = []
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            print("IS IMAGE STR: ", type(image))
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                print("THIS IS BYTE ARRAY")
            else:
                # if its a string
                print("THIS IS FLOAT STRING")
                image = torch.FloatTensor(image)

            # model specific pre-processing
            image = self.model_preprocess(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        with torch.no_grad():
            model_output = self.model(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        softmax_output = F.softmax(inference_output, dim=1)
        # print("softmax_output: ", softmax_output)
        return torch.argmax(softmax_output, dim = 1)

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.data_preprocess(data)
        print("Model Input: ", model_input.shape)
        model_output = self.inference(model_input)
        post_process_output = self.postprocess(model_output)
        print("post_process_output: ", post_process_output)
        return post_process_output.to("cpu").numpy().tolist()

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()


if __name__ == "__main__":

    class DotDict:
        def __init__(self, data):
            self.data = data

        def __getattr__(self, key):
            return self.data.get(key)

    # Example usage
    context = {"system_properties": {"gpu_id": 0, "model_dir": "./"}}
    context = DotDict(context)

    data = {"data": open("./0.png", "rb").read()}
    files = [data]
    handler = ModelHandler()
    handler.initialize(context)
    handler_output = handler.handle(files, context)
    print(type(handler_output))
    print("Handler Output: ", handler_output)
