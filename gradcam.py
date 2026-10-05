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
        
        # Register hooks and store handles for clean removal
        self.h1 = self.target_layer.register_forward_hook(self.save_activation)
        self.h2 = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        if hasattr(self, 'h1') and self.h1 is not None:
            self.h1.remove()
        if hasattr(self, 'h2') and self.h2 is not None:
            self.h2.remove()

    def __call__(self, input_tensor, target_category=None):
        self.model.eval()
        
        # Make tensor require grad for backward pass
        input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor_grad)

        if target_category is None:
            target_category = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_category] = 1
        output.backward(gradient=one_hot)

        if self.gradients is not None and self.activations is not None:
            gradients = self.gradients[0].cpu().data.numpy()
            activations = self.activations[0].cpu().data.numpy()

            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]

            cam = np.maximum(cam, 0)
            if cam.max() != 0:
                cam = cam / cam.max()
        else:
            cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
        
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        # Always remove hooks after computation to prevent hook accumulation
        self.remove()
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
