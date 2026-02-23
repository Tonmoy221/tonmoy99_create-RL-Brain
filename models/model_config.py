from typing_extensions import Literal, TypeAlias

from .wan_video_dit import WanModel
from .wan_video_dit_s2v import WanS2VModel
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_image_encoder import WanImageEncoder
from .wan_video_vae import WanVideoVAE, WanVideoVAE38
from .wan_video_motion_controller import WanMotionControllerModel
from .wan_video_vace import VaceWanModel
from .wav2vec import WanS2VAudioEncoder
from .wan_video_animate_adapter import WanAnimateAdapter
from .wan_video_mot import MotWanModel
from .longcat_video_dit import LongCatVideoTransformer3DModel

model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "2267d489f0ceb9f21836532952852ee5", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5ec04e02b42d2580483ad69f4e76346a", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5f90e66a0672219f12d9a626c8c21f61", ["wan_video_dit", "wan_video_vap"], [WanModel,MotWanModel], "diffusers"),
    (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "7a513e1f257a861512b1afd387a8ecd9", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "966cffdcc52f9c46c391768b27637614", ["wan_video_dit"], [WanS2VModel], "civitai"),
    (None, "8b27900f680d7251ce44e2dc8ae1ffef", ["wan_video_dit"], [LongCatVideoTransformer3DModel], "civitai"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
    (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
    (None, "06be60f3a4526586d8431cd038a71486", ["wans2v_audio_encoder"], [WanS2VAudioEncoder], "civitai"),
    (None, "31fa352acb8a1b1d33cd8764273d80a2", ["wan_video_dit", "wan_video_animate_adapter"], [WanModel, WanAnimateAdapter], "civitai"),
]
huggingface_model_loader_configs = [
    ("MarianMTModel", "transformers.models.marian.modeling_marian", "translator", None),
    ("BloomForCausalLM", "transformers.models.bloom.modeling_bloom", "beautiful_prompt", None),
    ("Qwen2ForCausalLM", "transformers.models.qwen2.modeling_qwen2", "qwen_prompt", None),
]
patch_model_loader_configs = [

]

preset_models_on_huggingface = {
    # RIFE
    "RIFE": [
        ("AlexWortega/RIFE", "flownet.pkl", "models/RIFE"),
    ],
}
preset_models_on_modelscope = {
    # RIFE
    "RIFE": [
        ("Damo_XR_Lab/cv_rife_video-frame-interpolation", "flownet.pkl", "models/RIFE"),
    ],
    # ESRGAN
    "ESRGAN_x4": [
        ("AI-ModelScope/Real-ESRGAN", "RealESRGAN_x4.pth", "models/ESRGAN"),
    ],
    # RIFE
    "RIFE": [
        ("AI-ModelScope/RIFE", "flownet.pkl", "models/RIFE"),
    ],
}
Preset_model_id: TypeAlias = Literal[
    "RIFE",
    "ESRGAN_x4",
    "RIFE",
]