import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_category=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_category is None:
            target_category = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_category] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() != 0:
            cam = cam / cam.max()
        
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam, target_category

def overlay_heatmap(original_img_rgb, cam_mask, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlays Grad-CAM mask over an RGB image (numpy uint8 arrays).
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure dimensions match
    if original_img_rgb.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
        
    overlay = cv2.addWeighted(original_img_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap
