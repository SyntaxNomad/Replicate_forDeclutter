import os
from typing import Optional
import torch
import numpy as np
import cv2
import tempfile
from PIL import Image
from cog import BasePredictor, Input, Path, Secret
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from transformers import pipeline as hf_pipeline


STYLES = {
    "bedroom": {
        "modern": {
            "prompt": (
                "A contemporary modern bedroom. Upholstered bed with curved headboard in warm grey velvet, "
                "sculptural bedside lamps, textured wall paneling, deep walnut wood floor, "
                "layered neutral textiles, ambient warm lighting, statement artwork above bed. "
                "Rich materials, considered design, no clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces "
            ),
        },
        "minimalist": {
            "prompt": (
                "A serene minimalist bedroom interior. Low platform bed with crisp white and warm ivory linen, "
                "natural light oak hardwood floor, soft warm white walls with subtle texture, "
                "one slim sculptural nightstand in pale wood, single architectural pendant light in matte white, "
                "clean uncluttered surfaces, soft natural light from large window. "
                "Warm, calm, breathable space. No decorations, no art, no plants, no rugs. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "scandinavian": {
            "prompt": (
                "A scandinavian hygge bedroom. Simple birch bed frame with white bedding, "
                "warm white walls, pine wood floor, soft pendant lamp, small plant. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "industrial": {
            "prompt": (
                "An industrial loft bedroom. Black metal bed frame, exposed brick wall, "
                "polished concrete floor, Edison pendant bulbs, black iron shelving. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "bohemian": {
            "prompt": (
                "A bohemian bedroom. Low wooden bed with layered terracotta textiles, "
                "woven rug, macrame wall hanging, brass pendant, trailing plants, earthy walls. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
    },
    "living_room": {
        "modern": {
            "prompt": (
                "A contemporary modern living room. Large curved sectional sofa in warm bouclé, "
                "sculptural marble coffee table, warm oak floor, textured plaster walls, "
                "dramatic pendant light, layered rugs, curated art and objects "
                "Rich and considered design, no clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "minimalist": {
            "prompt": (
                "A serene minimalist living room interior. Single low-profile sofa in warm ivory or soft greige linen, "
                "slim pale oak or walnut coffee table, soft warm white walls, light oak hardwood floor, "
                "one architectural floor lamp in matte white, clean uncluttered surfaces, "
                "large window with soft natural light flooding the room. "
                "Warm, calm, breathable space. No art, no plants, no rugs, no decorations. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "scandinavian": {
            "prompt": (
                "A scandinavian living room. Linen sofa, birch coffee table, white walls, "
                "oak floor, woolen throw, small plant. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "industrial": {
            "prompt": (
                "An industrial living room. Dark leather sofa, steel coffee table, "
                "concrete wall, Edison bulbs, metal shelving. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "bohemian": {
            "prompt": (
                "A bohemian living room. Colorful cushions, kilim rugs, macrame, rattan, "
                "trailing plants, string lights, terracotta walls. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
    },
    "bathroom": {
        "modern": {
            "prompt": (
                "A contemporary modern bathroom. Freestanding sculptural soaking tub, "
                "large format warm stone tiles, floating double vanity, brushed gold fixtures, "
                "backlit frameless mirror, warm ambient lighting, single large plant. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "minimalist": {
            "prompt": (
                "A serene minimalist bathroom interior. Floating white vanity with undermount basin, "
                "warm white plaster or large-format stone walls, light travertine or concrete floor, "
                "single large frameless mirror, brushed brass or matte black fixtures, "
                "recessed warm lighting, clean uncluttered surfaces, no toiletries visible. "
                "Warm, calm, spa-like atmosphere. No decorations, no plants. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "spa": {
            "prompt": (
                "A luxury spa bathroom. Travertine stone, deep soaking tub, rainfall shower, "
                "teak accents, ambient lighting, tropical plants. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "industrial": {
            "prompt": (
                "An industrial bathroom. Concrete walls, black fixtures, vessel sink, "
                "Edison mirror, black hexagon tiles. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
    },
    "kitchen": {
        "modern": {
            "prompt": (
                "A contemporary modern kitchen. Handleless cabinets in deep charcoal and warm oak, "
                "thick marble island with waterfall edge, statement pendant lights, "
                "integrated appliances, warm under-cabinet lighting, herringbone tile backsplash. "
                "Counters clear, considered design. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "minimalist": {
            "prompt": (
                "A serene minimalist kitchen interior. Seamless handleless cabinets in warm white or soft greige, "
                "thick stone or concrete countertop, fully integrated hidden appliances, "
                "warm under-cabinet lighting, light oak or concrete floor, "
                "all surfaces completely clear, no objects visible anywhere. "
                "Warm, calm, breathable space. No decorations, no plants. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "rustic": {
            "prompt": (
                "A rustic farmhouse kitchen. Shaker oak cabinets, butcher block counters, "
                "farmhouse sink, open shelving, exposed beams. Counters clear. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
        "industrial": {
            "prompt": (
                "An industrial kitchen. Stainless steel counters, black cabinets, "
                "exposed brick, metal shelving, Edison bulbs. Counters clear. "
                "Interior photography, sharp focus, 8k, photorealistic, ultra realistic photo, natural lighting, soft shadows, realistic textures, DSLR photography, 35mm lens, global illumination, physically based rendering, high detail surfaces"
            ),
        },
    },
}


class Predictor(BasePredictor):

    def setup(self):
        self.depth_estimator = None
        self.pipe = None
        self._models_loaded = False

    def _load_models(self):
        if self._models_loaded:
            return

        print("Loading depth estimator...")
        self.depth_estimator = hf_pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if torch.cuda.is_available() else -1,
        )

        print("Loading ControlNet Union Pro 2.0...")
        controlnet_union = FluxControlNetModel.from_pretrained(
            "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        controlnet = FluxMultiControlNetModel([controlnet_union])

        print("Loading FLUX schnell pipeline...")
        self.pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe.enable_attention_slicing()
        self._models_loaded = True

        print("Setup complete.")

    def predict(
        self,
        image: Path = Input(description="Photo of your room"),
        room_type: str = Input(
            description="Type of room",
            choices=["bedroom", "living_room", "bathroom", "kitchen"],
            default="bedroom",
        ),
        style: str = Input(
            description="Design style. bedroom/living_room: modern|minimalist|scandinavian|industrial|bohemian. bathroom: modern|minimalist|spa|industrial. kitchen: modern|minimalist|rustic|industrial",
            choices=["modern", "minimalist", "scandinavian", "industrial", "bohemian", "spa", "rustic"],
            default="minimalist",
        ),
        extra_prompt: str = Input(
            description="Optional extra details e.g. 'with a fireplace' or 'warm lighting'",
            default="",
        ),
        hf_token: Optional[Secret] = Input(
            description="Optional Hugging Face token (hf_...) for gated model access",
            default=None,
        ),
    ) -> Path:

        if hf_token is None:
            token_value = None
        elif isinstance(hf_token, str):
            token_value = hf_token
        else:
            token_value = hf_token.get_secret_value()

        if not self._models_loaded:
            if not token_value:
                raise ValueError("HuggingFace token is required for the first run")

            from huggingface_hub import login
            login(token=token_value)

            try:
                self._load_models()
            except Exception as e:
                raise RuntimeError(
                    "Model initialization failed. Check HF access or dependencies."
                ) from e

        if style not in STYLES[room_type]:
            available = list(STYLES[room_type].keys())
            raise ValueError(f"Style '{style}' not available for '{room_type}'. Choose from: {available}")

        input_image = Image.open(str(image)).convert("RGB").resize((512, 512), Image.LANCZOS)

        style_config = STYLES[room_type][style]
        prompt = style_config["prompt"]
        if extra_prompt.strip():
            prompt = prompt + " " + extra_prompt.strip() + "."
        negative_prompt = style_config.get("negative_prompt", "")

        print(f"Room: {room_type} | Style: {style}")

        # Depth map
        depth = self.depth_estimator(input_image)["depth"]
        depth_image = depth.convert("RGB").resize(input_image.size)

        # Generate
        result = self.pipe(
            prompt=prompt,
            control_image=[depth_image],
            control_mode=[2],
            controlnet_conditioning_scale=[0.8],
            control_guidance_end=0.8,
            num_inference_steps=8,
            guidance_scale=3.5,
            height=512,
            width=512,
        ).images[0]

        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        result.save(tmp)
        return Path(tmp)