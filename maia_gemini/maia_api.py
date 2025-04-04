from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline#, AutoPipelineForText2Image
import sys
# import random
import numpy as np
# import cv2
# import PIL
from PIL import Image
import PIL
import matplotlib.image as mpimg
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
import baukit #pip install git+https://github.com/davidbau/baukit
from baukit import Trace
import openai
import requests
from io import BytesIO
import sys
import io
from tqdm import tqdm

sys.path.append('./synthetic-neurons-dataset/')
sys.path.append('./synthetic-neurons-dataset/Grounded-Segment-Anything/')
sys.path.append('./netdissect/')
from netdissect.imgviz import ImageVisualizer 
from IPython import embed
import os
import base64
from typing import List, Tuple
from call_agent import ask_agent
import time
import math
sys.path.append('./synthetic-neurons-dataset/')
import synthetic_neurons
import clip
import torch.nn.functional as F
from SAE.sae import SparseAutoencoder9, SparseAutoencoder8, SparseAutoencoder6, SparseAutoencoder7


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

class System:
    """
    A Python class containing the vision model and the specific neuron to interact with.
    
    Attributes
    ----------
    neuron_num : int
        The serial number of the neuron.
    layer : string
        The name of the layer where the neuron is located.
    model_name : string
        The name of the vision model.
    model : nn.Module
        The loaded PyTorch model.
    neuron : callable
        A lambda function to compute neuron activation and activation map per input image. 
        Use this function to test the neuron activation for a specific image.
    device : torch.device
        The device (CPU/GPU) used for computations.

    Methods
    -------
    load_model(model_name: str)->nn.Module
        Gets the model name and returns the vision model from PyTorch library.
    call_neuron(image_list: List[torch.Tensor])->Tuple[List[int], List[str]]
        returns the neuron activation for each image in the input image_list as well as the activation map 
        of the neuron over that image, that highlights the regions of the image where the activations 
        are higher (encoded into a Base64 string).
    """



    def __init__(self, neuron_num: int, layer: str, model_name: str, device: str, thresholds=None, sae_bool=False):
        """
        Initializes a neuron object by specifying its number and layer location and the vision model that the neuron belongs to.
        Parameters
        -------
        neuron_num : int
            The serial number of the neuron.
        layer : str
            The name of the layer that the neuron is located at.
        model_name : str
            The name of the vision model that the neuron is part of.
        device : str
            The computational device ('cpu' or 'cuda').
        """
        self.neuron_num = neuron_num
        self.layer = layer
        self.device = torch.device("cuda") #torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.sae_bool = sae_bool
        self.sae = SparseAutoencoder8(input_channels=2048).to(self.device)
        self.LOAD_SAE_MODEL_FILE = '/home/nmital/datadrive/AutoInterpret/maia_gemini/results/trained_models/SAE_on[resnet152][layer=4][8][k_sparse]'
        load_checkpoint(torch.load(self.LOAD_SAE_MODEL_FILE + f".pth", map_location=torch.device('cpu')), self.sae)
        self.sae.to(self.device)

        self.preprocess = None
        if 'dino' in model_name or 'resnet' in model_name:
            self.preprocess = self.preprocess_imagenet
        self.model = self.load_model(model_name) #if clip, the define self.preprocess
        if thresholds is not None:
            self.threshold = thresholds[self.layer][self.neuron_num]
        else: 
            self.threshold = 0

    def load_model(self, model_name: str)->torch.nn.Module:
        """
        Gets the model name and returns the vision model from pythorch library.
        Parameters
        ----------
        model_name : str
            The name of the model to load.
        
        Returns
        -------
        nn.Module
            The loaded PyTorch vision model.
        
        Examples
        --------
        >>> # load "resnet152"
        >>> def execute_command(model_name) -> nn.Module:
        >>>   model = load_model(model_name: str)
        >>>   return model
        """
        if model_name=='resnet152':
            resnet152 = models.resnet152(weights='IMAGENET1K_V1').to(self.device)  
            model = resnet152.eval()
        elif model_name == 'dino_vits8':
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(self.device).eval()
        elif model_name == "clip-RN50": 
            name = 'RN50'
            full_model, preprocess = clip.load(name)
            model = full_model.visual.to(self.device).eval()
            self.preprocess = preprocess
        elif model_name == "clip-ViT-B32": 
            name = 'ViT-B/32'
            full_model, preprocess = clip.load(name)
            model = full_model.visual.to(self.device).eval()
            self.preprocess = preprocess
        return model

    
    @staticmethod
    def spatialize_vit_mlp(hiddens: torch.Tensor) -> torch.Tensor:
        """Make ViT MLP activations look like convolutional activations.
    
        Each activation corresponds to an image patch, so we can arrange them
        spatially. This allows us to use all the same dissection tools we
        used for CNNs.
    
        Args:
            hiddens: The hidden activations. Should have shape
                (batch_size, n_patches, n_units).
    
        Returns:
            Spatially arranged activations, with shape
                (batch_size, n_units, sqrt(n_patches - 1), sqrt(n_patches - 1)).
        """
        batch_size, n_patches, n_units = hiddens.shape
    
        # Exclude CLS token.
        hiddens = hiddens[:, 1:]
        n_patches -= 1
    
        # Compute spatial size.
        size = math.isqrt(n_patches)
        assert size**2 == n_patches
    
        # Finally, reshape.
        return hiddens.permute(0, 2, 1).reshape(batch_size, n_units, size, size)

    def calc_activations(self, image: torch.Tensor)->Tuple[int, torch.Tensor]:
        """"
        Returns the neuron activation for the input image, as well as the activation map of the neuron over the image
        that highlights the regions of the image where the activations are higher (encoded into a Base64 string).
    
        Parameters
        ----------
        image : torch.Tensor
            The input image in PIL format.
        
        Returns
        -------
        Tuple[int, torch.Tensor]
            Returns the maximum activation value of the neuron on the input image and a mask
        
        Examples
        --------
        >>> # load neuron 62, layer4 of resnet152
        >>> def execute_command(model_name) -> callable:
        >>>   model = load_model(model_name: str)
        >>>   neuron = load_neuron(neuron_num=62, layer='layer4', model=model)
        >>>   return neuron
        """
        with Trace(self.model, self.layer) as ret:
            _ = self.model(image)
            hiddens = ret.output
            if self.sae_bool:
                print("Inspecting the SAE")
                hiddens, reconstructed_feature = self.sae(hiddens) #

        if "dino" in self.model_name:
            hiddens = self.spatialize_vit_mlp(hiddens)

        batch_size, channels, *_ = hiddens.shape
        activations = hiddens.permute(0, 2, 3, 1).reshape(-1, channels)
        pooled, _ = hiddens.view(batch_size, channels, -1).max(dim=2)
        neuron_activation_map = hiddens[:, self.neuron_num, :, :]

        return(pooled[:,self.neuron_num], neuron_activation_map)
    
    def calc_class(self, image: torch.Tensor)->Tuple[int, torch.Tensor]:
        """"
        Returns the neuron activation for the input image, as well as the activation map of the neuron over the image
        that highlights the regions of the image where the activations are higher (encoded into a Base64 string).
    
        Parameters
        ----------
        image : torch.Tensor
            The input image in PIL format.
        
        Returns
        -------
        Tuple[int, torch.Tensor]
            Returns the maximum activation value of the neuron on the input image and a mask
        
        Examples
        --------
        >>> # load neuron 62, layer4 of resnet152
        >>> def execute_command(model_name) -> callable:
        >>>   model = load_model(model_name: str)
        >>>   neuron = load_neuron(neuron_num=62, layer='layer4', model=model)
        >>>   return neuron
        """
        logits = self.model(image)
        prob = F.softmax(logits, dim=1)
        image_calss = torch.argmax(logits[0])
        activation = prob[0][image_calss]
        return activation.item(), image

    def call_neuron(self, image_list: List[torch.Tensor])->Tuple[List[int], List[str]]:
        """
        The function returns the neuron’s maximum activation value (in int format) over each of the images in the list as well as the activation map of the neuron over each of the images that highlights the regions of the image where the activations are higher (encoded into a Base64 string).
        
        Parameters
        ----------
        image_list : List[torch.Tensor]
            The input image
        
        Returns
        -------
        Tuple[List[int], List[str]]
            For each image in image_list returns the maximum activation value of the neuron on that image, and a masked images, 
            with the region of the image that caused the high activation values highlighted (and the rest of the image is darkened). Each image is encoded into a Base64 string.

        
        Examples
        --------
        >>> # test the activation value of the neuron for the prompt "a dog standing on the grass"
        >>> def execute_command(system, prompt_list) -> Tuple[int, str]:
        >>>     prompt = ["a dog standing on the grass"]
        >>>     image = text2image(prompt)
        >>>     activation_list, activation_map_list = system.call_neuron(image)
        >>>     return activation_list, activation_map_list
        >>> # test the activation value of the neuron for the prompt “a fox and a rabbit watch a movie under a starry night sky” “a fox and a bear watch a movie under a starry night sky” “a fox and a rabbit watch a movie at sunrise”
        >>> def execute_command(system.neuron, prompt_list) -> Tuple[int, str]:
        >>>     prompt_list = [[“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]]
        >>>     images = text2image(prompt_list)
        >>>     activation_list, activation_map_list = system.call_neuron(images)
        >>>     return activation_list, activation_map_list
        """
        activation_list = []
        masked_images_list = []
        input_image_list = []
        for image in image_list:
            if  image==None: #for dalle
                activation_list.append(None)
                masked_images_list.append(None)
            else:
                if self.layer == 'last':
                    tensor = self.preprocess_images(image)
                    acts, image_class = self.calc_class(tensor)    
                    activation_list.append(torch.round(acts[ind] * 100).item()/100)
                    masked_images_list.append(image2str(image[0]))
                    input_image_list.append(img2str(image[0]))
                else:
                    image = self.preprocess_images(image)
                    acts,masks = self.calc_activations(image)    
                    ind = torch.argmax(acts).item()
                    masked_image = generate_masked_image(image[ind], masks[ind], "./temp.png", self.threshold)
                    input_image = generate_masked_image(image[ind], torch.ones_like(masks[ind]), "./temp.png", torch.zeros_like(self.threshold))
                    activation_list.append(torch.round(acts[ind] * 100).item()/100)   
                    masked_images_list.append(masked_image)
                    input_image_list.append(input_image)

        return activation_list, input_image_list, masked_images_list
    
    def preprocess_imagenet(self, image, normalize=True, im_size=224):
        
        if normalize:
            preprocess = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
            ])
        return preprocess(image)
        

    def preprocess_images(self, images):
        image_list = []
        if type(images) == list:
            for image in images:
                image_list.append(self.preprocess(image).to(self.device))
            batch_tensor = torch.stack(image_list)
            return batch_tensor
        else:
            return self.preprocess(images).unsqueeze(0).to(self.device)


class Synthetic_System:
    """
    A Python class containing the vision model and the specific neuron to interact with.
    
    Attributes
    ----------
    neuron_num : int
        The serial number of the neuron.
    layer : string
        The name of the layer where the neuron is located.
    model_name : string
        The name of the vision model.
    model : nn.Module
        The loaded PyTorch model.
    neuron : callable
        A lambda function to compute neuron activation and activation map per input image. 
        Use this function to test the neuron activation for a specific image.
    device : torch.device
        The device (CPU/GPU) used for computations.

    Methods
    -------
    load_model(model_name: str)->nn.Module
        Gets the model name and returns the vision model from PyTorch library.
    call_neuron(image_list: List[torch.Tensor])->Tuple[List[int], List[str]]
        returns the neuron activation for each image in the input image_list as well as the activation map 
        of the neuron over that image, that highlights the regions of the image where the activations 
        are higher (encoded into a Base64 string).
    """
    def __init__(self, neuron_num: int, neuron_labels: str, neuron_mode: str, device: str):
        
        self.neuron_num = neuron_num
        self.neuron_labels = neuron_labels
        self.neuron = synthetic_neurons.SAMNeuron(neuron_labels, neuron_mode)
        self.device = torch.device("cuda") #torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.threshold = 0
        self.layer = neuron_mode


    def call_neuron(self, image_list: List[torch.Tensor])->Tuple[List[int], List[str]]:
        """
        The function returns the neuron’s maximum activation value (in int format) over each of the images in the list as well as the activation map of the neuron over each of the images that highlights the regions of the image where the activations are higher (encoded into a Base64 string).
        
        Parameters
        ----------
        image_list : List[torch.Tensor]
            The input image
        
        Returns
        -------
        Tuple[List[int], List[str]]
            For each image in image_list returns the maximum activation value of the neuron on that image, and a masked images, 
            with the region of the image that caused the high activation values highlighted (and the rest of the image is darkened). Each image is encoded into a Base64 string.

        
        Examples
        --------
        >>> # test the activation value of the neuron for the prompt "a dog standing on the grass"
        >>> def execute_command(system, prompt_list) -> Tuple[int, str]:
        >>>     prompt = ["a dog standing on the grass"]
        >>>     image = text2image(prompt)
        >>>     activation_list, activation_map_list = system.call_neuron(image)
        >>>     return activation_list, activation_map_list
        >>> # test the activation value of the neuron for the prompt “a fox and a rabbit watch a movie under a starry night sky” “a fox and a bear watch a movie under a starry night sky” “a fox and a rabbit watch a movie at sunrise”
        >>> def execute_command(system.neuron, prompt_list) -> Tuple[int, str]:
        >>>     prompt_list = [[“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]]
        >>>     images = text2image(prompt_list)
        >>>     activation_list, activation_map_list = system.call_neuron(images)
        >>>     return activation_list, activation_map_list
        """
        activation_list = []
        masked_images_list = []
        for image in image_list:
            if  image==None: #for dalle
                activation_list.append(None)
                masked_images_list.append(None)
            else:
                acts, _, _, masks = self.neuron.calc_activations(image)    
                ind = np.argmax(acts)
                masked_image = image2str(masks[ind], "./temp_synthetic.png")
                activation_list.append(acts[ind])
                masked_images_list.append(masked_image)
        return activation_list,masked_images_list

class Tools:
    """
    A Python class containing tools to interact with the neuron implemented in the system class, 
    in order to run experiments on it.
    
    Attributes
    ----------
    text2image_model_name : str
        The name of the text-to-image model.
    text2image_model : any
        The loaded text-to-image model.
    images_per_prompt : int
        Number of images to generate per prompt.
    path2save : str
        Path for saving output images.
    threshold : any
        Activation threshold for neuron analysis.
    device : torch.device
        The device (CPU/GPU) used for computations.
    experiment_log: str
        A log of all the experiments, including the code and the output from the neuron

    Methods
    -------
    text2image(prompt_list: str)->Tuple[torcu.Tensor]
        Gets a list of text prompt as an input and generates an image for each prompt in the list using a text to image model.
        The function returns a list of images.
    load_text2image_model(model_name: str) -> any
        Loads a text-to-image model.
    text2image(prompt: str) -> List[any]
        Generates images based on a text prompt.
    sampler(act: any, imgs: List[any], mask: any, prompt: str, method: str = 'max') -> Tuple[List[int], List[str]]
        Processes images based on neuron activations.
    generate_masked_image(image: any, mask: any, path2save: str) -> str
        Generates a masked image highlighting high activation areas.
    preprocess_image(self, images: any, normalize: bool = True) -> torch.Tensor
        Preprocesses images for the model.
    describe_images(image_list: List[str], image_title:List[str], desctiprions = List[str]) -> str
        Gets a list of images and generat a textual description of the unmasked regions within each of them.
    
    """

    def __init__(self, path2save, device, DatasetExemplars = None, images_per_prompt=10, text2image_model_name='sd'):
        """
        Initializes the Tools object.

        Parameters
        ----------
        path2save : str
            Path for saving output images.
        DatasetExemplars : object
            an object from the class DatasetExemplars
        device : str
            The computational device ('cpu' or 'cuda').
        """
        self.device = torch.device("cuda") #torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.text2image_model_name = text2image_model_name
        self.text2image_model = self.load_text2image_model(model_name=text2image_model_name)
        self.images_per_prompt = images_per_prompt
        self.p2p_model_name = 'ip2p'
        self.p2p_model = self.load_pix2pix_model(model_name=self.p2p_model_name) # consider maybe adding options for other models like pix2pix zero
        self.path2save = path2save
        self.experiment_log = []
        self.im_size = 224
        if DatasetExemplars is not None:
            self.exemplars = DatasetExemplars.exemplars
            self.masked_exemplars = DatasetExemplars.masked_exemplars  
            self.exemplars_activations = DatasetExemplars.activations
            self.exempalrs_thresholds = DatasetExemplars.thresholds
        self.activation_threshold = 0
        self.results_list = []


    def text2image(self, prompt_list: List[str]) -> List[torch.Tensor]:
        """Gets a list of text prompt as an input, generates an image for each prompt in the list using a text to image model.
        The function returns a list of images.

        Parameters
        ----------
        prompt_list : List[str]
            A list of text prompts for image generation.

        Returns
        -------
        List[Image.Image]
            A list of images, corresponding to each of the input prompts. 


        Examples
        --------
        >>> # test the activation value of the neuron for the prompt "a dog standing on the grass"
        >>> def execute_command(system, tools) -> Tuple[int, str]:
        >>>     prompt = ["a dog standing on the grass"]
        >>>     image = tools.text2image(prompt)
        >>>     activation_list, activation_map_list = system.call_neuron(image)
        >>>     return activation_list, activation_map_list
        >>> # test the activation value of the neuron for the prompt “a fox and a rabbit watch a movie under a starry night sky” “a fox and a bear watch a movie under a starry night sky” “a fox and a rabbit watch a movie at sunrise”
        >>> def execute_command(system.neuron, tools) -> Tuple[int, str]:
        >>>     prompt_list = [[“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]]
        >>>     images = tools.text2image(prompt_list)
        >>>     activation_list, activation_map_list = system.call_neuron(images)
        >>>     return activation_list, activation_map_list
        """
        image_list = [] 
        for prompt in prompt_list:
            while True:
                try:
                    #with torch.amp.autocast(device_type='mps'):
                    images = self.prompt2image(prompt)
                    break
                except Exception as e:
                    print(e)
            image_list.append(images)
        return image_list
    
    def edit_images(self, image_prompt_list_org : List[Image.Image], editing_instructions_list : List[str], batch_size=32):
        """Gets a list of prompts to generate images, and list of corresponding editing prompts as an input, edits each image based on the instructions given in the prompt using a text-based image editing model.
        Important note: Do not use negative terminology such as "remove ...", try to use terminology like "replace ... with ..." or "change the color of ... to"
        The function returns a list of images.

        Parameters
        ----------
        image_prompt_list_org : List[Image.Image]
            A list of input ptompts to generate images according to, these images are to be edited by the prompts in editing_instructions_list.
        editing_instructions_list : List[str]
            A list of instructions for how to edit the images in image_list. Should be the same length as image_list.

        Returns
        -------
        List[Image.Image], List[str]
            A list of images, corresponding to each of the input images and corresponding editing prompts
            and a list of all the prompts that were used in the experiment, in the same order as the images

        Examples
        --------
        >>> # test the activation value of the neuron for the prompt "a dog standing on the grass" and test the effect of changing the dog to a cat
        >>> def execute_command(system, prompt_list) -> Tuple[int, str]:
        >>>     prompt = ["a dog standing on the grass"]
        >>>     edits = ["replace the dog with a cat"]
        >>>     images, images_edited = edit_images(prompt, edits)
        >>>     activation_list, activation_map_list = system.call_neuron(images + images_edited)
        >>>     return activation_list, activation_map_list
        """
        image_list = []
        for prompt in image_prompt_list_org:
            #with torch.amp.autocast(device_type='cpu'):
            image_list.append(self.prompt2image(prompt, images_per_prompt=1)[0])
        image_list = [item for item in image_list if item is not None]
        editing_instructions_list = [item for item, condition in zip(editing_instructions_list, image_list) if condition is not None]
        image_prompt_list_org = [item for item, condition in zip(image_prompt_list_org, image_list) if condition is not None]

        edited_images = self.p2p_model(editing_instructions_list, image_list).images
        all_images= []
        all_prompt = []
        for i in range(len(image_prompt_list_org)*2):
            if i%2 == 0:
                all_prompt.append(image_prompt_list_org[i//2])
                all_images.append([image_list[i//2]])
            else:
                all_prompt.append(editing_instructions_list[i//2])
                all_images.append([edited_images[i//2]])
        return all_images, all_prompt
    
    def save_experiment_log(self, activation_list: List[int], image_list: List[str], masked_image_list: List[str], image_titles: List[str], image_textual_information: List[str] = None):
        """documents the current experiment results as an entry in the experiment log list. if self.activation_threshold was updated by net_dissect function, 
        the experiment log will contains instruction to continue with experiments if activations are lower than activation_threshold.
        Results that are loged will be available for future experiment (unlogged results will be unavailable).
        The function also update the attribure "result_list", such that each element in the result_list is a dictionary of the format: {"<prompt>": {"activation": act, "image": image}}
        so the list contains all the resilts that were logged so far.

        Parameters
        ----------
        activation_list : List[int]
            A list of the activation values that were achived for each of the images in "image_list".
        image_list : List[str]
            A list of the images that were generated using the text2image model and were tested.
        image_titles : List[str]
            A list of the text prompts that were tested. according to these prompt the images in "image_list" were generated.
        additional_information: (Union[str, List[str]])
            A string or a list of additional text to log
        
        Returns
        -------
            None

        Examples
        --------
        >>> # tests the activation value of the neuron for the prompt "a dog standing on the grass" and logs 
        >>> def execute_command(System.neuron, prompt_list) -> Tuple[int, str]:
        >>>     prompt = ["a dog standing on the grass"]
        >>>     activation_list, activation_map_list = tools.text2activation(System.neuron, prompt)
        >>>     save_experiment_log(prompt, activation_list, activation_map_list)
        >>> # tests the activation value of the neuron for the prompts “a fox and a rabbit watch a movie under a starry night sky” “a fox and a bear watch a movie under a starry night sky” “a fox and a rabbit watch a movie at sunrise” and logs all results
        >>> def execute_command(System.neuron, prompt_list) -> Tuple[int, str]:
        >>>     prompt_list = [“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]
        >>>     activation_list, activation_map_list = text2activation(System.neuron, prompt_list)
        >>>     save_experiment_log(prompt_list, activation_list, activation_map_list)
        >>> # tests the activation value of the neuron for the prompts “a fox and a rabbit watch a movie under a starry night sky” “a fox and a bear watch a movie under a starry night sky” “a fox and a rabbit watch a movie at sunrise” and logs the results and the image descriptions 
        >>> def execute_command(system, tools):
        >>>     prompt_list = [“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]
        >>>     images = tools.text2image(prompt_list)
        >>>     activation_list, activation_map_list = system.call_neuron(images)
        >>>     descriptions = describe_images(images, prompt_list)
        >>>     save_experiment_log(prompt_list, activation_list, activation_map_list, descriptions)
        >>>     return 
        >>> # tests network dissect exemplars and logs the results and the image descriptions 
        >>> def execute_command(system, tools):
        >>>     activation_list, image_list = self.net_dissect(system)
        >>>     prompt_list = []
        >>>     for i in range(len(activation_list)):
        >>>          prompt_list.append(f'network dissection, exemplar {i}') # for the network dissection exemplars e don't have prompts, therefore need to provide text titles
        >>>     descriptions = describe_images(image_list, prompt_list)
        >>>     save_experiment_log(prompt_list, activation_list, activation_map_list, descriptions)
        >>>     return 
        >>> # tests the activation value of the neuron for the prompt “a fox and a rabbit watch a movie under a starry night sky” “a fox and a bear watch a movie under a starry night sky” “a fox and a rabbit watch a movie at sunrise” but only logs the result with the highest activation 
        >>> def execute_command(System.neuron, prompt_list) -> Tuple[int, str]:
        >>>     prompt_list = [“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]
        >>>     activation_list, activation_map_list = text2activation(System.neuron, prompt_list)
        >>>     max_ind = torch.argmax(act).item()
        >>>     save_experiment_log(prompt_list[max_ind], activation_list[max_ind], activation_map_list[max_ind])
        >>> # tests 10 different prompts and logs 5 result with the highest activation 
        >>> def execute_command(System.neuron, prompt_list) -> Tuple[int, str]:
        >>>     prompt_list = [“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”, ...]
        >>>     activation_list, activation_map_list = text2activation(System.neuron, prompt_list)
        >>>     sorted_values, indices = torch.sort(activation_list)
        >>>     save_experiment_log(prompt_list[indices[-5:]], activation_list[indices[-5:]], activation_map_list[indices[-5:]])
        >>> # tests 10 different prompts and logs only results that got activations higher than a defined threshold
        >>> def execute_command(System.neuron, prompt_list) -> Tuple[int, str]:
        >>>     prompt_list = [“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”, ...]
        >>>     activation_list, activation_map_list = text2activation(System.neuron, prompt_list)
        >>>     threshold = THRESHOLD #defined by the user
        >>>     save_experiment_log(prompt_list[activation_list > THRESHOLD], activation_list[activation_list > THRESHOLD], activation_map_list[activation_list > THRESHOLD])
        """
        output = [{"type":"text", "text": 'Neuron activations:\n'}]
        for ind,act in enumerate(activation_list):
            output.append({"type": "text", "text": f'"{image_titles[ind]}", activation: {act}\nimage: \n'})
            #output.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_list[ind]}})
            output.append({"type": "image_url", "image_url": image_list[ind]})
            output.append({"type": "image_url", "image_url": masked_image_list[ind]})
            self.results_list.append({image_titles[ind]: {"activation": act, "image": image_list[ind], "masked_image": masked_image_list[ind]}})
        if (self.activation_threshold != 0) and (max(activation_list) < self.activation_threshold):
            output.append({"type": "text", "text":f"\nMax activation is smaller than {(self.activation_threshold * 100).round()/100}, please continue with the experiments.\n"})
        if image_textual_information != None:
            if isinstance(image_textual_information, list):
                for text in image_textual_information:
                    output.append({"type": "text", "text": text})
            else:
                output.append({"type": "text", "text": image_textual_information})
        self.update_experiment_log(role='user', content=output)

    def update_experiment_log(self, role, content=None, type=None, type_content=None):
        openai_role = {'execution':'user','maia':'assistant','user':'user','system':'system'}
        if type == None:
            self.experiment_log.append({'role': openai_role[role], 'content': content})
        elif content == None:
            if type == 'text':
                self.experiment_log.append({'role': openai_role[role], 'content': [{"type":type, "text": type_content}]})
                # self.experiment_log.append({'role': openai_role[role], 'content': {"type":type, "text": type_content}}) #gemini
            if type == 'image_url':
                self.experiment_log.append({'role': openai_role[role], 'content': [{"type":type, "image_url": type_content}]}) #gemini
                # self.experiment_log.append({'role': role, 'content': {"type":type, "image_url": {'url': type_content}}}) #gemini
    
    def dataset_exemplars(self, system):
        """
        Retrieves the activation and exemplar image list for a specific neuron in a given layer.

        This method accesses stored data for a specified neuron within a layer of the neural network. 
        It returns both the activation values and the corresponding exemplar images that were used 
        to generate these activations. The neuron and layer are specified through a 'system' object.

        Parameters
        ----------
        system : System
            An object representing the specific neuron and layer within the neural network.
            The 'system' object should have 'layer' and 'neuron_num' attributes.

        Returns
        -------
        tuple
            A tuple containing three elements:
            - The first element is a list of activation values for the specified neuron.
            - The second element is a list of exemplar images (as Base64 encoded strings or 
            in the format they were stored) corresponding to these activations.
            - The third element is a list of masked exemplar images (as Base64 encoded strings or 
            in the format they were stored) masked with the activations.

        Example
        -------
        >>> def execute_command(system, tools)
        >>> activation_list, image_list = tools.net_dissect(system_instance)
        >>> prompt_list = []
        >>> for i in range(len(activation_list)):
        >>>     prompt_list.append(f'network dissection, exemplar {i}')
        >>> save_experiment_log(prompt_list, activation_list, image_list, self.activation_threshold)
        """
        image_list = self.exemplars[system.layer][system.neuron_num]
        masked_image_list = self.masked_exemplars[system.layer][system.neuron_num]
        activation_list = self.exemplars_activations[system.layer][system.neuron_num]
        self.activation_threshold = sum(activation_list)/len(activation_list)
        activation_list = (activation_list * 100).round()/100 
        return activation_list, image_list, masked_image_list

    def neuron_exemplars(self, system):
        """
        Retrieves the activation and exemplar neuron list for a specific image in a given layer.

        This method accesses stored data for a specified neuron within a layer of the neural network.
        It returns both the activation values and the corresponding exemplar neurons that fire for the given image.
        The neuron and layer are specified through a 'system' object.

        Parameters
        ----------
        system : System
            An object representing the specific neuron and layer within the neural network.
            The 'system' object should have 'layer' and 'neuron_num' attributes.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - The first element is a list of activation values for the specified neuron.
            - The second element is a list of exemplar images (as Base64 encoded strings or
            in the format they were stored) corresponding to these activations.

        Example
        -------
        >>> def execute_command(system, tools)
        >>> activation_list, image_list = tools.net_dissect(system_instance)
        >>> prompt_list = []
        >>> for i in range(len(activation_list)):
        >>>     prompt_list.append(f'network dissection, exemplar {i}')
        >>> save_experiment_log(prompt_list, activation_list, image_list, self.activation_threshold)
        """
        image_list = self.exemplars[system.layer][system.neuron_num]
        activation_list = self.exemplars_activations[system.layer][system.neuron_num]
        self.activation_threshold = sum(activation_list)/len(activation_list)
        activation_list = (activation_list * 100).round()/100
        return activation_list, image_list

    def load_pix2pix_model(self, model_name):
        """
        Loads a pix2pix image editing model.

        Parameters
        ----------
        model_name : str
            The name of the pix2pix model.

        Returns
        -------
        The loaded pix2pix model.
        """
        if model_name == "ip2p": # instruction tuned pix2pix model
            device = self.device
            model_id = "timbrooks/instruct-pix2pix"
            #pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
            #torch.mps.set_per_process_memory_fraction(0.0)
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            return pipe
        else:
            raise(ValueError("unrecognized pix2pix model name"))

    
    def load_text2image_model(self,model_name):
        """
        Loads a text-to-image model.

        Parameters
        ----------
        model_name : str
            The name of the text-to-image model.

        Returns
        -------
        The loaded text-to-image model.
        """
        if model_name == "sd":
            device = self.device
            model_id = "runwayml/stable-diffusion-v1-5"
            #sdpipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            sdpipe = StableDiffusionPipeline.from_pretrained(model_id)
            #torch.mps.set_per_process_memory_fraction(0.0)
            sdpipe = sdpipe.to(device)
            sdpipe.enable_attention_slicing()
            return sdpipe
        elif model_name == "sdxl-turbo":
            device = self.device
            model_id = "stabilityai/sdxl-turbo"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
            #pipe = AutoPipelineForText2Image.from_pretrained(model_id)
            #torch.mps.set_per_process_memory_fraction(0.0)
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
            return pipe
        elif model_name == "dalle":
            pipe = None
            return pipe
        else:
            raise(ValueError("unrecognized text to image model name"))
    
    def prompt2image(self, prompt, images_per_prompt=None):
        if images_per_prompt == None: images_per_prompt = self.images_per_prompt
        if self.text2image_model_name == "sd":
            if images_per_prompt > 1:
                prompts = [prompt] * images_per_prompt
            else: prompts = prompt
            #with torch.amp.autocast(device_type='cpu'):
            images = self.text2image_model(prompts).images
        elif self.text2image_model_name == "sdxl-turbo":
            if images_per_prompt > 1:
                prompts = [prompt] * images_per_prompt
            else: prompts = prompt
            #with torch.amp.autocast(device_type='cpu'):
            images = self.text2image_model(prompt=prompts, num_inference_steps=4, guidance_scale=0.0).images
        elif self.text2image_model_name == "dalle":
            if images_per_prompt > 1:
                raise(ValueError("cannot use DALLE with 'images_per_prompt'>1 due to rate limits"))
            else:
                images = []
                try:
                    response = openai.Image.create(prompt=prompt, n=1, size="256x256")
                except Exception as e:
                    print(e)
                    return images.append(None)
                image_url = response["data"][0]["url"]
                response = requests.get(image_url)
                # Check if the request was successful (status code 200)
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
                images.append(image)
        else:
            raise(ValueError("unrecognized text to image model name"))
        return images
        
    
    def sampler(self, act, imgs, mask, prompt, threshold, method = 'max'):  
        if method=='max':
            max_ind = torch.argmax(act).item()
            masked_image_max = self.generate_masked_image(image=imgs[max_ind],mask=mask[max_ind],path2save=f"{self.path2save}/{prompt}_masked_max.png", threshold=threshold)
            acts= max(act).item()
            ims = masked_image_max
        elif method=='min_max':
            max_ind = torch.argmax(act).item()
            min_ind = torch.argmin(act).item()
            masked_image_max = self.generate_masked_image(image=imgs[max_ind],mask=mask[max_ind],path2save=f"{self.path2save}/{prompt}_masked_max.png", threshold=threshold)
            masked_image_min = self.generate_masked_image(image=imgs[min_ind],mask=mask[min_ind],path2save=f"{self.path2save}/{prompt}_masked_min.png", threshold=threshold)
            acts = []
            ims = [] 
            acts.append(max(act).item())
            acts.append(min(act).item())
            ims.append(masked_image_max)
            ims.append(masked_image_min)
        return acts, ims
 
    def summarize_images(self, image_list: List[str]) -> str:
        """
        Gets a list of images and describes what is common to all of them, focusing specifically on unmasked regions.


        Parameters
        ----------
        image_list : list
            A list of images in Base64 encoded string format.
        
        Returns
        -------
        str
            A string with a descriptions of what is common to all the images.

        Example
        -------
        >>> # tests dataset dissect exemplars and logs the results and the image descriptions 
        >>> def execute_command(system, tools):
        >>>     activation_list, image_list = self.dataset_dissect(system)
        >>>     prompt_list = []
        >>>     for i in range(len(activation_list)):
        >>>          prompt_list.append(f'network dissection, exemplar {i}') # for the network dissection exemplars e don't have prompts, therefore need to provide text titles
        >>>     summarization = tools.summarize_images(image_list)
        >>>     save_experiment_log(prompt_list, activation_list, activation_map_list, summarization)
        >>>     return 

        """
        instructions = "What do all the unmasked regions of these images have in common? There might be more then one common concept, or a few groups of images with different common concept each. In these cases return all of the concepts.. Return your description in the following format: [COMMON]: <your description>."
        #history = [{'role':'system', 'content':'you are an helpful assistant'}]
        history = [{'role': 'system', 'content': [{"type": 'text', 'text': 'you are a helpful assistant'}]}]
        user_contet = [{"type":"text", "text": instructions}]
        for ind,image in enumerate(image_list):
            #user_contet.append({"type": "image_url", "image_url": {"data:image/jpeg;base64," + image}, "details": "high"})
            user_contet.append({"type": "image_url", "image_url": image})
        history.append({'role': 'user', 'content': user_contet})
        #description = ask_agent('gpt-4-vision-preview',history)
        description = ask_agent('gemini-1.5-flash', history)
        if isinstance(description, Exception): return description
        return description

    def describe_images(self, image_list: List[str], image_title:List[str]) -> str:
        """
        Generates descriptions for a list of images, focusing specifically on highlighted regions.

        This function iterates through a list of images, requesting a description for the 
        highlighted (unmasked) regions in each image. The final descriptions are concatenated 
        and returned as a single string, with each description associated with the corresponding 
        image title.

        Parameters
        ----------
        image_list : list
            A list of images in Base64 encoded string format.
        image_title : callable
            A function or lambda that takes an index (integer) and returns a corresponding 
            title (string) for each image.

        Returns
        -------
        str
            A concatenated string of descriptions for each image, where each description 
            is associated with the image's title and focuses on the highlighted regions 
            in the image.

        Example
        -------
        >>> def execute_command(system, tools):
        >>>     prompt_list = [“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]
        >>>     images = tools.text2image(prompt_list)
        >>>     activation_list, activation_map_list = system.call_neuron(images)
        >>>     descriptions = describe_images(activation_map_list, prompt_list)
        >>>     return descriptions
        >>> def execute_command(system, tools):
        >>>     activation_list, image_list = self.net_dissect(system)
        >>>     prompt_list = []
        >>>     for i in range(len(activation_list)):
        >>>          prompt_list.append(f'network dissection, exemplar {i}') # for the network dissection exemplars e don't have prompts, therefore need to provide text titles
        >>>     descriptions = describe_images(image_list, prompt_list)
        >>>     return descriptions

        """
        description_list = ''
        instructions = "Do not describe the full image. Please describe ONLY the unmasked regions in this image (e.g. the regions that are not darkened). Be as concise as possible. Return your description in the following format: [highlighted regions]: <your concise description>"
        # time.sleep(60)
        #history = [{'role': 'system', 'content': [{'text': 'you are a helpful assistant'}]}]
        #user_contet = [{"type": "text", "text": instructions}]
        for ind, image in enumerate(image_list):
            # user_contet.append({"type": "image_url", "image_url": {"data:image/jpeg;base64," + image}, "details": "high"})
            #user_contet.append({"type": "image_url", "image_url": image})
            '''for ind,image in enumerate(image_list):
            history = [{'role':'system', 'content':'you are an helpful assistant'},{'role': 'user', 'content': [{"type":"text", "text": instructions}, {"type": "image_url", "image_url": "data:image/jpeg;base64," + image}]}]'''
            history = [{'role': 'system', 'content': [{'type': 'text', 'text': 'you are a helpful assistant'}]}, {'role': 'user', 'content': [
                {"type": "text", "text": instructions},
                {"type": "image_url", "image_url": image}]}]
            #description = ask_agent('gpt-4-vision-preview',history)
            description = ask_agent('gemini-1.5-flash', history)
            if isinstance(description, Exception): return description_list
            description = description.split("[highlighted regions]:")[-1]
            description = " ".join([f'"{image_title[ind]}", highlighted regions:',description])
            description_list += description + '\n'
        return description_list

    def generate_html(self,name="experiment.html"):
        # Generates an HTML file with the experiment log.
        html_string = f'''<html>
        <head>
        <title>Experiment Log</title>
        <!-- Include Prism Core CSS (Choose the theme you prefer) -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css" rel="stylesheet" />
        <!-- Include Prism Core JavaScript -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/prism.min.js"></script>
        <!-- Include the Python language component for Prism -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-python.min.js"></script>
        </head> 
        <body>
        <h1>{self.path2save}</h1>'''

        # don't plot system+user prompts (uncomment if you want the html to include the system+user prompts)
        '''
        html_string += f"<h2>{self.experiment_log[0]['role']}</h2>"
        html_string += f"<pre><code>{self.experiment_log[0]['content']}</code></pre><br>"
        
        html_string += f"<h2>{self.experiment_log[1]['role']}</h2>"
        html_string += f"<pre>{self.experiment_log[1]['content'][0]}</pre><br>"
        initial_images = ''
        initial_activations = ''
        for cont in self.experiment_log[1]['content'][1:]:
            if isinstance(cont, dict):
                initial_images += f"<img src="data:image/png;base64,{cont['image']}"/>"
            else:
                initial_activations += f"{cont}    "
        html_string +=  initial_images
        html_string += f"<p>Activations:</p>"
        html_string += initial_activations
        '''

        for entry in self.experiment_log[2:]:      
            if entry['role'] == 'assistant':
                html_string += f"<h2>MAIA</h2>"  
                html_string += f"<pre>{entry['content'][0]['text']}</pre><br>"
            else:
                html_string += f"<h2>Experiment Execution</h2>"  
                for content_entry in entry['content']:

                    if "image_url" in content_entry["type"]:
                        #html_string += f'''<img src="{content_entry['image_url']['url']}"/>'''
                        html_string += f'''<img src= 'data:image/jpeg;base64, {content_entry['image_url']}'/>'''
                    elif "text" in content_entry["type"]:
                        html_string += f"<pre>{content_entry['text']}</pre>"
        html_string += '</body></html>'

        file_html = open(os.path.join(self.path2save, name), "w")
        file_html.write(html_string)
        file_html.close()

def generate_masked_image(image,mask,path2save,threshold):
    #Generates a masked image highlighting high activation areas.
    vis = ImageVisualizer(224, image_size=224, source='imagenet')
    masked_tensor = vis.pytorch_masked_image(image, activations=mask, unit=None, level=threshold, outside_bright=0.25) #percent_level=0.95)
    masked_image = Image.fromarray(masked_tensor.permute(1, 2, 0).byte().cpu().numpy())
    buffer = BytesIO()
    masked_image.save(buffer, format="PNG")
    buffer.seek(0)
    masked_image = base64.b64encode(buffer.read()).decode('ascii')
    return(masked_image)

def image2str(image,path2save):
    #Converts an image to a Base64 encoded string.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode('ascii')
    return(image)

def str2image(image_str):
    #Converts a Base64 encoded string to an image.
    img_bytes = base64.b64decode(image_str)
    img_buffer = BytesIO(img_bytes)
    img = Image.open(img_buffer)
    return img

class DatasetExemplars():
    """
    A class for performing network dissection on a given neural network model.

    This class analyzes specific layers and units of a neural network model to 
    understand what each unit in a layer has learned. It uses a set of exemplar 
    images to activate the units and stores the resulting activations, along with
    visualizations of the activated regions in the images.

    Attributes
    ----------
    path2exemplars : str
        Path to the directory containing the exemplar images.
    n_exemplars : int
        Number of exemplar images to use for each unit.
    path2save : str
        Path to the directory where the results will be saved.
    model_name : str
        Name of the neural network model being dissected.
    layers : list
        List of layer names in the model to be dissected.
    units : list
        List of unit indices to be analyzed. If None, all units are analyzed.
    im_size : int, optional
        Size to which the images will be resized (default is 224).

    Methods
    -------
    net_dissect(layer: str, im_size: int=224)
        Dissects the specified layer of the neural network, analyzing the response
        to the exemplar images and saving visualizations of activated regions.
    """

    def __init__(self, path2exemplars, path2save, model_name, layers, units, n_exemplars = 15, im_size=224, sae_bool=False, compute_exemplars=False, device='cpu'):

        """
        Constructs all the necessary attributes for the DatasetExemplars object.

        Parameters
        ----------
        path2exemplars : str
            Path to the directory containing the exemplar images.
        n_exemplars : int
            Number of exemplar images to use for each unit.
        path2save : str
            Path to the directory where the results will be saved.
        model_name : str
            Name of the neural network model being dissected.
        layers : list
            List of layer names in the model to be dissected.
        units : list
            List of unit indices to be analyzed. If None, all units are analyzed.
        im_size : int, optional
            Size to which the images will be resized (default is 224).
        """

        self.path2exemplars = path2exemplars
        self.n_exemplars = n_exemplars
        self.path2save = path2save
        self.model_name = model_name
        self.layers = layers
        self.units = units
        self.im_size = im_size
        self.sae_bool = sae_bool
        self.compute_exemplars = compute_exemplars
        self.device = device
        self.exemplars = {}
        self.masked_exemplars = {}
        self.activations = {}
        self.thresholds = {}
        self.model = self.load_model(model_name)
        self.sae = SparseAutoencoder8(input_channels=2048).to(self.device)
        self.LOAD_SAE_MODEL_FILE = '/home/nmital/datadrive/AutoInterpret/maia_gemini/results/trained_models/SAE_on[resnet152][layer=4][8][k_sparse]'
        load_checkpoint(torch.load(self.LOAD_SAE_MODEL_FILE + f".pth", map_location=torch.device('cpu')), self.sae)
        self.sae.to(self.device)
        if isinstance(self.layers, list):
            for layer in self.layers: 
                exemplars, masked_exemplars, activations, thresholds = self.net_dissect(layer)
                self.exemplars[layer] = exemplars
                self.masked_exemplars[layer] = masked_exemplars
                self.activations[layer] = activations
                self.thresholds[layer] = thresholds
        else:
            exemplars, masked_exemplars, activations, thresholds = self.net_dissect(self.layers)
            self.exemplars[self.layers] = exemplars
            self.masked_exemplars[self.layers] = masked_exemplars
            self.activations[self.layers] = activations
            self.thresholds[self.layers] = thresholds

    def net_dissect(self,layer,im_size=224):

        """
        Dissects the specified layer of the neural network.

        This method analyzes the response of units in the specified layer to
        the exemplar images. It generates and saves visualizations of the
        activated regions in these images.

        Parameters
        ----------
        layer : str
            The name of the layer to be dissected.
        im_size : int, optional
            Size to which the images will be resized for visualization 
            (default is 224).

        Returns
        -------
        tuple
            A tuple containing lists of images, activations, and thresholds 
            for the specified layer. The images are Base64 encoded strings.
        """
        with torch.no_grad():
            if not self.compute_exemplars:
                print('stored exemplars')
                exp_path = f'{self.path2exemplars}/{self.model_name}/imagenet/{layer}'
                activations = np.loadtxt(f'{exp_path}/activations.csv', delimiter=',')
                thresholds = np.loadtxt(f'{exp_path}/thresholds.csv', delimiter=',')
                image_array = np.load(f'{exp_path}/images.npy')
                mask_array = np.load(f'{exp_path}/masks.npy')
                all_images = []
                all_masked_images = []
                for unit in range(activations.shape[0]):
                    curr_image_list = []
                    if self.units!=None and not(unit in self.units):
                        all_images.append(curr_image_list)
                        continue
                    for exemplar_inx in range(min(activations.shape[1],self.n_exemplars)):
                        save_path = os.path.join(self.path2save,'dataset_exemplars',self.model_name,layer,str(unit),'netdisect_exemplars')
                        if os.path.exists(os.path.join(save_path,f'{exemplar_inx}.png')):
                            with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                                masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                            curr_image_list.append(masked_image)
                        else:
                            curr_mask = np.repeat(mask_array[unit,exemplar_inx], 3, axis=0)
                            curr_image = image_array[unit,exemplar_inx]
                            inside = np.array(curr_mask>0)
                            outside = np.array(curr_mask==0)
                            masked_image = curr_image * inside + 0.25 * curr_image * outside
                            ''' Original input image '''
                            input_image =  Image.fromarray(np.transpose(curr_image, (1, 2, 0)).astype(np.uint8))
                            input_image = input_image.resize([self.im_size, self.im_size], Image.Resampling.LANCZOS)
                            os.makedirs(save_path,exist_ok=True)
                            input_image.save(os.path.join(save_path,f'orig_{exemplar_inx}.png'), format='PNG')
                            with open(os.path.join(save_path,f'orig_{exemplar_inx}.png'), "rb") as image_file:
                                input_image = base64.b64encode(image_file.read()).decode('utf-8')
                            ''' Masked image '''
                            masked_image =  Image.fromarray(np.transpose(masked_image, (1, 2, 0)).astype(np.uint8))
                            masked_image = masked_image.resize([self.im_size, self.im_size], Image.Resampling.LANCZOS)
                            os.makedirs(save_path,exist_ok=True)
                            masked_image.save(os.path.join(save_path,f'{exemplar_inx}.png'), format='PNG')
                            with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                                masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                            curr_image_list.append(masked_image)
                            input_image_list.append(input_image)
                    all_images.append(input_image_list)
                    all_masked_images.append(curr_image_list)

            else:
                print('computing exemplars')
                dataset_path = '/home/nmital/datadrive/AutoInterpret/maia_gemini/SAE/dataset/imagenet-mini/'
                self.val_dataset, self.val_loader = self.get_dataloader(dataset_path, mode="test")
                loop = tqdm(self.val_loader, leave=True)
                thresholds = None

                for idx, (images, _) in enumerate(loop):
                    images = images.to(self.device)
                    #with torch.autocast(device_type=self.device):
                    with Trace(self.model, self.layers) as ret:
                        _ = self.model(images)
                        hiddens = ret.output
                    if self.sae_bool:
                        hiddens, reconstructed_feature = self.sae(hiddens.float())  #
                    all_images = [[] for _ in range(hiddens.shape[1])]
                    all_masked_images = [[] for _ in range(hiddens.shape[1])]
                    hidden_activations_on_units = hiddens[:, self.units[0], :, :].clone() #.squeeze(1)
                    #hiddens_max_units = torch.max(hiddens.flatten(start_dim=2, end_dim=3), dim=2).values >= torch.quantile(hiddens, 0.95) # sae activations in the 95th percentile. Useful if you just want to find the neurons which activate the most.
                    hiddens_on_units = torch.max(hiddens.flatten(start_dim=2, end_dim=3), dim=2).values
                    if thresholds is None:
                        thresholds = torch.quantile(hiddens.permute(1,0,2,3).flatten(start_dim=1, end_dim=3), 0.85, dim=1)
                    else:
                        t = torch.quantile(hiddens.permute(1,0,2,3).flatten(start_dim=1, end_dim=3), 0.85, dim=1)
                        thresholds = torch.where(t > thresholds, t, thresholds)
                    if idx>0:
                        hidden_activations_on_units = torch.cat((hidden_activations_on_units, exemplars_activation_maps), dim=0)
                        hiddens_on_units = torch.cat((hiddens_on_units, exemplars_batch.values), dim=0)
                        images = torch.cat((images, exemplars_images), dim=0)

                    exemplars_batch = torch.topk(hiddens_on_units, self.n_exemplars, dim=0)
                    exemplars_images = images[exemplars_batch.indices[:, self.units[0]]]
                    exemplars_activation_maps = hidden_activations_on_units[exemplars_batch.indices[:, self.units[0]]]

                print("Saving SAE exemplars")
                activations = torch.transpose(exemplars_batch.values, 0, 1)
                for ind in range(self.n_exemplars):
                    save_path = os.path.join(self.path2save, f'dataset_exemplars_sae[{self.sae_bool}]', self.model_name, layer, str(self.units[0]), 'netdisect_exemplars')
                    masked_image = generate_masked_image(exemplars_images[ind, ...], exemplars_activation_maps[ind, ...], "./temp.png", thresholds[self.units[0]])
                    ''' Original input images '''
                    input_image = generate_masked_image(exemplars_images[ind, ...], torch.ones_like(exemplars_activation_maps[ind, ...]), "./temp.png", torch.zeros_like(thresholds[self.units[0]]))
                    i = base64.b64decode(input_image)
                    i = io.BytesIO(i)
                    i = mpimg.imread(i, format='PNG')
                    os.makedirs(save_path, exist_ok=True)
                    im = Image.fromarray((i*255).astype(np.uint8))
                    im.save(os.path.join(save_path, f'orig_{ind}.png'))
                    all_images[self.units[0]].append(input_image)
                    ''' Masked_image '''
                    i = base64.b64decode(masked_image)
                    i = io.BytesIO(i)
                    i = mpimg.imread(i, format='PNG')
                    os.makedirs(save_path, exist_ok=True)
                    im = Image.fromarray((i*255).astype(np.uint8))
                    im.save(os.path.join(save_path, f'{ind}.png'))
                    all_masked_images[self.units[0]].append(masked_image)

        return all_images,all_masked_images,activations[:,:self.n_exemplars],thresholds
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""

        transform = transforms.Compose([
            transforms.Resize([224,224]),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(dataset_path, 'val'),
            transform=transform)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=6,
            drop_last=True)

        return dataset, data_loader  # Return both dataset and loader

    def load_model(self, model_name: str) -> torch.nn.Module:
        """
        Gets the model name and returns the vision model from pythorch library.
        Parameters
        ----------
        model_name : str
            The name of the model to load.

        Returns
        -------
        nn.Module
            The loaded PyTorch vision model.

        Examples
        --------
        >>> # load "resnet152"
        >>> def execute_command(model_name) -> nn.Module:
        >>>   model = load_model(model_name: str)
        >>>   return model
        """
        if model_name == 'resnet152':
            resnet152 = models.resnet152(weights='IMAGENET1K_V1').to(self.device)
            model = resnet152.eval()
        elif model_name == 'dino_vits8':
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(self.device).eval()
        elif model_name == "clip-RN50":
            name = 'RN50'
            full_model, preprocess = clip.load(name)
            model = full_model.visual.to(self.device).eval()
            self.preprocess = preprocess
        elif model_name == "clip-ViT-B32":
            name = 'ViT-B/32'
            full_model, preprocess = clip.load(name)
            model = full_model.visual.to(self.device).eval()
            self.preprocess = preprocess
        return model

class SyntheticExemplars():
    
    def __init__(self, path2exemplars, path2save, mode, n_exemplars=15, im_size=224):
        self.path2exemplars = path2exemplars
        self.n_exemplars = n_exemplars
        self.path2save = path2save
        self.im_size = im_size
        self.mode = mode

        self.exemplars = {}
        self.activations = {}
        self.thresholds = {}

        exemplars, activations = self.net_dissect()
        self.exemplars[mode] = exemplars
        self.activations[mode] = activations

    def net_dissect(self,im_size=224):
        exp_path = f'{self.path2exemplars}/{self.mode}/'
        activations = np.loadtxt(f'{exp_path}/activations.csv', delimiter=',')
        image_array = np.load(f'{exp_path}/images.npy')
        mask_array = np.load(f'{exp_path}/masks.npy')
        all_images = []
        for unit in range(activations.shape[0]):
            curr_image_list = []
            for exemplar_inx in range(min(activations.shape[1],self.n_exemplars)):
                save_path = os.path.join(self.path2save,'synthetic_exemplars',self.mode,str(unit))
                if os.path.exists(os.path.join(save_path,f'{exemplar_inx}.png')):
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(masked_image)
                else:
                    curr_mask = (mask_array[unit,exemplar_inx]/255)+0.25
                    curr_mask[curr_mask>1]=1
                    curr_image = image_array[unit,exemplar_inx]
                    masked_image = curr_image * curr_mask
                    masked_image =  Image.fromarray(masked_image.astype(np.uint8))
                    os.makedirs(save_path,exist_ok=True)
                    masked_image.save(os.path.join(save_path,f'{exemplar_inx}.png'), format='PNG')
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(masked_image)
            all_images.append(curr_image_list)

        return all_images,activations[:,:self.n_exemplars]
