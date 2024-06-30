import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor
myh=0
myw=0
class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al. 
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''
    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers
        
        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)
        
        self.activations = []
        self.grads = []
        
    def forward_hook(self, module, input, output):
        #print(len(input)) #1
        #print(len(output)) #40
        #bb
        # print("#input:",input[0].shape)#input: torch.Size([40, 512, 7, 7])
        # print("#module:",module)
        # for i in range(input[0].shape[0]):
        #     output_o = output[i]#.cpu().data.numpy().squeeze()
        #     # print("#output[0]:",output[0].shape)#output[0]: torch.Size([512, 7, 7])
        #     # print("#output_o:",output_o.shape)
        #     #print("#output_d:",output_o.detach())
        #     #print("#output_d:",output_o.detach().shape)
        #     self.activations.append(output_o)
        self.activations = []
        self.activations.append(output)

    def backward_hook(self, module, grad_input, grad_output):
        #print("#grad_output[0]:",grad_output[0].shape)#grad_output[0]: torch.Size([40, 512, 7, 7])
        self.grads = []
        self.grads.append(grad_output[0].detach())
        
    def calculate_cam(self, model_input):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)                 # Module.to() is in-place method 
            model_input = model_input.to(device)  # Tensor.to() is not a in-place method
        self.model.eval()
        
        self.model.zero_grad()
        # forward
        # print("#model_input:",model_input.shape)#model_input: torch.Size([40, 3, 224, 224])
        y_hat = self.model(model_input)
        # print("#y_hat:",y_hat.shape)#y_hat: torch.Size([40, 1000])
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)
        # print("#max_class:",max_class)
        # print("#max_class_shape:",max_class.shape)#max_class_shape: (40,)
        # backward
        
        y_c = y_hat[:, max_class]
        # print("#y_c:",y_c)
        # print("#y_c_shape:",y_c.shape)#y_c_shape: torch.Size([40])
        y_c.backward(torch.ones_like(y_c))
        
        # get activations and gradients
        #list_shape = self.activations.cpu().numpy().shape
        #activations_o = self.activations.cpu().data.numpy()
        # print("#self_activations_shape:",activations_o.shape)
        # print("#activations[0]:",self.activations[0].cpu().shape)#activations[0]: torch.Size([512, 7, 7])
        activations = self.activations[0]#self.activations#.cpu().data.numpy().squeeze()#activations_shape: (512, 7, 7)
        #cam=np.expand_dims(activations,axis=2)
        #print("#self_activations_shape:",activations_o.shape)
        #print(activations)
        grads = self.grads[0]#.cpu().data.numpy().squeeze()
        # print("#grads_shape:",grads.shape)#grads_shape: (40, 512, 7, 7)
        # for iii in range(grads.shape[0]):
        #     print(iii,torch.sum(grads[iii]))
        # bb
        # calculate weights
        tmp = grads.reshape(grads.shape[0],grads.shape[1], -1)
        #print(tmp.shape) (512, 49)
        weights = torch.mean(tmp, axis=-1)
        weights = weights.reshape(grads.shape[0],grads.shape[1], 1, 1)
        # print("#weights_shape:",weights.shape)
        
        # print("#activations_shape:",activations.shape)#activations_shape: (512, 7, 7)
        # bb
        #grads_shape: (40, 512, 7, 7)
        #weights_shape: (512, 1, 1)  -> 40, 512, 1, 1
        #activations_shape: (512, 7, 7) -> 40, 512, 7, 7
        #train_img: torch.Size([40, 3, 224, 224])
        #cam = (weights * activations).sum(axis=0)
        cam = (weights * activations).sum(axis=1)
        #cam = torch.clamp(cam,0)
        cam[cam < 0] = 0
        cam = cam.view(cam.shape[0], 1, cam.shape[1], cam.shape[2])
        #print(cam.shape) 
        cam = torch.nn.functional.interpolate(
                cam, size=(224, 224), mode='bilinear', align_corners=True)
        # cam = cam.cpu().data.numpy()
        # print(cam.shape) 
        # bb
        # cam = np.maximum(cam, 0) # ReLU 相当于把小于0的置0
        # cam = cam / cam.max()
        # cam = cv2.resize(cam, (224,224))
        cam = cam / cam.max()
        #print(cam.shape)
        
        return cam
    
    @staticmethod
    def show_cam_on_image(image, cam):
        # image: [H,W,C]
        h, w = image.shape[:2]
        
        cam = cv2.resize(cam, (h,w))
        cam = cam / cam.max()
        print("cam:",cam.shape)
        #cam.expand_dims(2)
        cam=np.expand_dims(cam,axis=2)
        #print("cam:",cam.shape)
        heatmap = cv2.applyColorMap((255*cam).astype(np.uint8), cv2.COLORMAP_JET) # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        image = image / image.max()
        heatmap = heatmap / heatmap.max()
        
        result = 0.4*heatmap + 0.6*image
        result = result / result.max()
        final=(result*255).astype(np.uint8)
        out=np.concatenate((image,cam),axis=2)
        #print("final:",final.shape)
        #print("out:",out.shape)
        cv2.imwrite('CAM_666.jpg', final)
        
        global myh,myw
        #print("myh:",myh)
        #print("myw:",myw)
        finalpic = cv2.resize(final, (myw,myh))
        plt.figure()
        plt.imshow(finalpic)
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        preprocessing = Compose([
        	ToTensor(),
        	Normalize(mean=mean, std=std)
            ])
        return preprocessing(img.copy()).unsqueeze(0) 


if __name__ == '__main__':
    img = cv2.imread('people.png') 
    print("image:",img.shape)
    # global myh,myw
    myh, myw = img.shape[:2]
    print("h:",myh)
    print("w:",myw)
    image = cv2.resize(img, (224,224))
    print("image:",image.shape)# (224,224,3)
    input_tensor = GradCAM.preprocess_image(image)
    print("#input_tensor:",input_tensor.shape)#input_tensor: torch.Size([1, 3, 224, 224])
    model = models.resnet18(pretrained=True)
    grad_cam = GradCAM(model, model.layer4[-1], 224)
    cam = grad_cam.calculate_cam(input_tensor)
    GradCAM.show_cam_on_image(image, cam)