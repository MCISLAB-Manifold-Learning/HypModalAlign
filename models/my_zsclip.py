import torch
import torch.nn as nn

class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.clip_model.eval()

        from .clip import tokenize
        self.tokenize = tokenize
        
    def encode_image(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def encode_text(self, texts):
        with torch.no_grad():
            # Tokenize文本
            tokenized_texts = self.tokenize(texts).cuda()
            text_features = self.clip_model.encode_text(tokenized_texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def __call__(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        
        return {'logits_per_image': logits_per_image, 'logits_per_text': logits_per_text}
        
    def compute_similarity_for_topk(self, images, texts_list, topk_indices):
        
        batch_size, k = topk_indices.shape
        
        image_features = self.encode_image(images)  # [batch_size, embed_dim]
        
        flat_indices = topk_indices.flatten()  # [batch_size * k]
        
        selected_texts = [texts_list[idx.item()] for idx in flat_indices]
        
        text_features = self.encode_text(selected_texts)  # [batch_size * k, embed_dim]
        
        text_features = text_features.view(batch_size, k, -1)
        

        image_features = image_features.unsqueeze(1)
        
        scores = (image_features * text_features).sum(dim=-1) * self.clip_model.logit_scale.exp()
        
        return scores

