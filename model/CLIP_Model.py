import sys
from CLIP import clip
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Image_Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        
        self.head = nn.Sequential(
            nn.Linear(512, 1)
        )
        
        self.img_token_proj = nn.Sequential(
            nn.Linear(768, 512)
        )
        
        self.patch_head = nn.Sequential(
            nn.Linear(512, 1)
        )
        self.text_linear = nn.Sequential(
            nn.Linear(512, 768)
        )
    
    def forward(self, x: torch.Tensor,text:torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) +\
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        n = x.shape[0]
        text = self.text_linear(text).unsqueeze(0).expand(n, -1, -1)
        x = torch.cat([x,text], dim=1)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        img_tokens = x[:, 1:, :]
        x = self.visual.ln_post(x[:, 0, :])
        
        if self.visual.proj is not None:
            x = x @ self.visual.proj
            
        return x, img_tokens
        
class ImagEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        
        self.head = nn.Sequential(
            nn.Linear(512, 1)
        )
        
        self.img_token_proj = nn.Sequential(
            nn.Linear(768, 512)
        )
        
        self.patch_head = nn.Sequential(
            nn.Linear(512, 1)
        )
        self.text_linaer = nn.Sequential(
            nn.Linear(512, 768)
        )

    def forward(self,image):
        cls_token, img_tokens = self.visual(image.type(self.dtype))
        patch_feat = self.img_token_proj(img_tokens)
        return cls_token, patch_feat


class Text_Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # print(tokenized_prompts.shape)
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x
    
class Text_Prompt(nn.Module):
    def __init__(self, model, initials=None):
        super(Text_Prompt,self).__init__()
        print("The initial prompts are:",initials)
        self.text_encoder = Text_Encoder(model)
        
        if isinstance(initials,list):
            text = clip.tokenize(initials)
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_())
            self.num_prompts = self.embedding_prompt.shape[0]
        elif isinstance(initials,str):
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            print([" ".join(["X"]*\
                16)," ".join(["X"]*16)])
            self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*\
                16)," ".join(["X"]*16)]).requires_grad_())).cuda()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self):
        tokenized_prompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*16)]])
        text_features = self.text_encoder(self.embedding_prompt,tokenized_prompts)
        return text_features

class MultiHeadAttention(nn.Module):
    def __init__(self, nin, n_head=2) -> None:
        super().__init__()
        self.nin = nin
        self.n_head = n_head
        self.k_embed = nn.Linear(self.nin, self.nin)
        self.q_embed = nn.Linear(self.nin, self.nin)
        self.v_embed = nn.Linear(self.nin, self.nin)
        self.fc = nn.Linear(self.nin, self.nin)

    def forward(self, x):
        B, N, C = x.shape
        key = self.k_embed(x)
        queue = self.q_embed(x)
        value = self.v_embed(x)

        key = key.view(B, -1, self.n_head, self.nin //
                       self.n_head).permute(0, 2, 1, 3)
        
        queue = queue.view(B, -1, self.n_head, self.nin //
                           self.n_head).permute(0, 2, 1, 3)
        value = value.view(B, -1, self.n_head, self.nin //
                           self.n_head).permute(0, 2, 1, 3)
        scale = torch.sqrt(torch.tensor(
            [self.nin // self.n_head], device=x.device))
        
        attention = torch.matmul(queue, key.permute(0, 1, 3, 2)) / scale
        attention = torch.softmax(attention, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, -1, self.nin)
        x = self.fc(x)
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class TransformerEncoderBlock1(nn.Sequential):
    '''
    输入x=torch.size([batch_size, N , 2*E])
    q与k,v不同
    '''
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(emb_size)
        self.feedforward = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        
    def forward(self, x):
        x1 = x[:,:577,:]
        x2 = x[:,577:,:]
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        x, _ = self.attention(x1, x2, x2)
        # print(x.shape)
        x = self.norm(x)
        x = self.feedforward(x)
        return x
    
class TransformerEncoderBlock1_224(nn.Sequential):
    '''
    输入x=torch.size([batch_size, N , 2*E])
    q与k,v不同
    '''
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(emb_size)
        self.feedforward = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        
    def forward(self, x):
        x1 = x[:,:197,:]
        x2 = x[:,197:,:]
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        x, _ = self.attention(x1, x2, x2)
        # print(x.shape)
        x = self.norm(x)
        x = self.feedforward(x)
        return x

class TransformerEncoderBlock2(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class VitFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x
    
from PIL import Image
class Face_Clip(nn.Module):
    def __init__(self, model, initials=None):
        super(Face_Clip,self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoderx = Text_Prompt(model,
            [
            "fake X X X X X X X X X X X X X X X",
            "X fake X X X X X X X X X X X X X X",
            "X X fake X X X X X X X X X X X X X",
            "X X X fake X X X X X X X X X X X X",
            "X X X X fake X X X X X X X X X X X",
            "X X X X X fake X X X X X X X X X X",
            "X X X X X X fake X X X X X X X X X",
            "X X X X X X X fake X X X X X X X X",
            "X X X X X X X X fake X X X X X X X",
            "X X X X X X X X X fake X X X X X X",
            "X X X X X X X X X X fake X X X X X",
            "X X X X X X X X X X X fake X X X X",
            "X X X X X X X X X X X X fake X X X",
            "X X X X X X X X X X X X X fake X X",
            "X X X X X X X X X X X X X X fake X",
            "X X X X X X X X X X X X X X X fake",
            "The forgery type of this fake face is Deepfakes",
            "The forgery type of this fake face is NeuralTextures",
            "The forgery type of this fake face is FaceSwap",
            "The forgery type of this fake face is Face2Face",
            "The forgery type of this fake face is X X X X",
            ]).to(device)
        
        set_requires_grad(self.text_encoderx.text_encoder, False)
        self.image_encoder = Image_Encoder(model)
        self.parameter = 4.0
        self.decoder = nn.Sequential(
            TransformerEncoderBlock2(),
            TransformerEncoderBlock2(),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.headx1 = nn.Sequential(
            nn.Linear(768, 512),
        )
        self.head1 = nn.Sequential(
            nn.Linear(768, 1),
        )
        self.headx2 = nn.Sequential(
            nn.Linear(768, 512),
        )
        self.head2 = nn.Sequential(
            nn.Linear(768, 1),
        )
        self.headx3 = nn.Sequential(
            nn.Linear(768, 512),
        )
        self.head3 = nn.Sequential(
            nn.Linear(768, 1),
        )
        self.headx4 = nn.Sequential(
            nn.Linear(768, 512),
        )
        self.head4 = nn.Sequential(
           nn.Linear(768, 1),
        )
        self.head5 = nn.Sequential(
           nn.Linear(768, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 1),
        )
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=2, batch_first=True)
        self.final = nn.Linear(5, 5, bias=True)
        self.type_linear = nn.Sequential(
            nn.Linear(512, 768)
        )
        self.weight = nn.Parameter(torch.ones(16))

    def forward(self, x):
        text_feature = self.text_encoderx()
        text_features = text_feature[0:16]
        type = text_feature[16:]
        
        image_feature, patch = self.image_encoder(x, type)
        type1 = patch[:,196,:] # torch.Size([batch, 768]) "The forgery type for this fake image is Deepfake",
        type2 = patch[:,197,:] # torch.Size([batch, 768]) "The forgery type for this fake image is NeuralTextures"
        type3 = patch[:,198,:] # torch.Size([batch, 768]) "The forgery type for this fake image is Face2Face"
        type4 = patch[:,199,:] # torch.Size([batch, 768]) "The forgery type for this fake image is FaceSwap"
        type5 = patch[:,200,:] # torch.Size([batch, 768]) "The forgery type for this fake image is X X X X"
       
        cls1 = self.head1(type1)
        cls2 = self.head2(type2)
        cls3 = self.head3(type3)
        cls4 = self.head4(type4)
        cls5 = self.head5(type5)
        
        text_type_feature = self.type_linear(type)
        feat = patch[:,:196,:]
        residual_feat = self.decoder(feat)
        residual_feat = residual_feat/residual_feat.norm(dim=-1, keepdim=True)
        residual_feat = residual_feat@(text_type_feature.T)
        residual_feat = self.resiadual_unpatchify(residual_feat)
        residual = F.softmax(residual_feat, dim=1)
        bin_tensor = torch.cat([cls1,cls2,cls3,cls4,cls5],dim=1)
        residual = residual * bin_tensor.unsqueeze(-1).unsqueeze(-1)
        residual = residual.sum(1, keepdim=True)
        for i in range(image_feature.shape[0]):
            image_features=image_feature[i]
            image_features = image_features / image_features.norm(dim=0, keepdim=True)
            nor=torch.norm(text_features,dim=-1, keepdim=True)
            
            similarity = (image_features @ (text_features/nor).T)
            similarity = self.weight * similarity
            similarity = torch.sum(similarity).view(1)/self.parameter
            if(i==0):
                probs=similarity
            else:
                probs=torch.cat([probs,similarity],dim=0)
        cls_real = 0
        return probs, cls_real, cls1, cls2, cls3, cls4, residual
    
    def test_forward(self, x):
        text_feature = self.text_encoderx()
        text_features = text_feature[0:16]
        type = text_feature[16:]
        image_feature, patch = self.image_encoder(x, type)
        for i in range(image_feature.shape[0]):
            image_features=image_feature[i]
            image_features = image_features / image_features.norm(dim=0, keepdim=True)
            nor=torch.norm(text_features,dim=-1, keepdim=True)
            similarity = (image_features @ (text_features/nor).T)
            similarity = self.weight * similarity
            similarity = torch.sum(similarity).view(1)/self.parameter
            if(i==0):
                probs=similarity
            else:
                probs=torch.cat([probs,similarity],dim=0)
        return probs#, image_feature#, cls, cls1, cls2, cls3, cls4
    
    def resiadual_unpatchify(self, x):
        """
        x: (N, T, 4)
        imgs: (N, H, W, 4)
        """
        c = 5
        p = 1
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        # c = self.out_channels
        c = 3
        # p = self.x_embedder.patch_size[0]
        p = 16
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    
if __name__ == '__main__':
    from CLIP import clip
    clip_model, preprocess = clip.load("ViT-B/16", \
    device=torch.device("cpu"), download_root="../model/clip_model")#ViT-B/16
    length_prompt = 16
    x = torch.randn([2, 3, 224, 224]).cuda()
    model = Face_Clip(clip_model)
    probs, cls, cls1, cls2, cls3, cls4 ,residual_feat= model(x)

    
