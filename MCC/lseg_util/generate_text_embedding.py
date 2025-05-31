import torch
import clip

class_name = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                'table', 'door', 'window', 'bookshelf', 'picture','counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
                'toilet', 'sink', 'bathtub', 'other', 'ceiling')

labelset = [ "a " + label + " in a scene" for label in class_name]
labelset[-2] = 'other'

labels = []
for line in labelset:
    label = line
    labels.append(label)

clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
# clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cuda', jit=False)


text = clip.tokenize(labels)
text = text.cuda()
text_features = clip_pretrained.encode_text(text)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
text_features = text_features.float()

text_features = text_features.cpu().detach()

torch.save(text_features, 'vocabulary_embedding.pth')



