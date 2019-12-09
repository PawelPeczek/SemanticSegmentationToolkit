from src.model.predefined.ic_net import ICNet
from src.model.predefined.ic_net_autoencoder import ICNetAutoEncoder
from src.model.predefined.ic_net_v12 import ICNetV12

NAME_TO_MODEL = {
    'ICNet': ICNet,
    'ICNetAutoEncoder': ICNetAutoEncoder,
    'ICNetV12': ICNetV12
}
