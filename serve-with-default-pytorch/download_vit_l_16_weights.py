import torch
import torchvision

# load model
model = torchvision.models.vit_l_16(weights = "ViT_L_16_Weights.DEFAULT")
print("Model is loaded.")
# preprocessing
preprocess = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()
preprocess.antialias = True

# save the state dict
torch.save(model.state_dict(), "vit_l_16.pt")
print("Model saved")
