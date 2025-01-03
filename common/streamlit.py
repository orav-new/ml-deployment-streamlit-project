from common.consts import PRETRAINED_RESNET_MODEL_PATH


import streamlit as st
import torch
from torchvision.models import resnet18


@st.cache_resource
def load_model(device: str = "cpu") -> torch.nn.Module:
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 27)  # EMNIST has 26 letters
    model.load_state_dict(torch.load(PRETRAINED_RESNET_MODEL_PATH, weights_only=False, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    return model