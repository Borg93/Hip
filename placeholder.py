# NOTE: This is the original napkin sketch of the idea. It is kept for reference
# but is SUPERSEDED by the package in src/hiptr/. The load id below is actually
# correct (TIPSv2 *is* a HF AutoModel via the -dpt repos), but you must use
# dpt._backbone.vision_encoder and its 3-tuple forward (there is no
# .last_hidden_state); labels must cover the spliced visual tokens; <loc_*> must
# be added to the tokenizer; normalization is ToTensor/[0,1] (the /255 here is
# fine); and 1024px is not divisible by the patch size (14). See DESIGN.md §7 for
# the full point-by-point list and how the new code fixes each issue.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import os

# --- 1. MODEL ARCHITECTURE ---
class FullHTRSystem(nn.Module):
    def __init__(self, vision_id="google/tipsv2-l14-dpt", llm_id="Qwen/Qwen3.5-0.8B"):
        super().__init__()
        
        # Vision Backbone (High-Res TIPSv2)
        print(f"Loading Vision: {vision_id}")
        self.vision_model = AutoModel.from_pretrained(vision_id, trust_remote_code=True)
        for param in self.vision_model.parameters():
            param.requires_grad = True # Full Fine-Tuning enabled
            
        # Language Decoder (Qwen 3.5)
        print(f"Loading Decoder: {llm_id}")
        self.decoder = AutoModelForCausalLM.from_pretrained(
            llm_id, 
            torch_dtype=torch.bfloat16, 
            device_map='auto', 
            attn_implementation="sdpa"
        )
        for param in self.decoder.parameters():
            param.requires_grad = True # Full Fine-Tuning enabled

        # The Bridge: Projects 1024 visual features to 1024 LLM space
        self.projector = nn.Linear(1024, 1024)

    def forward(self, pixel_values, input_ids, labels=None):
        # Vision feature extraction
        vis_feats = self.vision_model(pixel_values).last_hidden_state
        
        # Project to LLM tokens
        vis_tokens = self.projector(vis_feats)
        
        # Text embeddings
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        
        # Concatenate and pass to decoder
        combined = torch.cat([vis_tokens, text_embeds], dim=1)
        return self.decoder(inputs_embeds=combined, labels=labels, return_dict=True).loss

# --- 2. DATA PIPELINE (ALTO XML) ---
class ALTOParserDataset(Dataset):
    def __init__(self, img_dir, xml_dir, tokenizer, max_length=1024):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.xml_paths = sorted(glob.glob(os.path.join(xml_dir, '*.xml')))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def parse_alto(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#'}
        page = root.find('.//alto:Page', ns)
        pw, ph = float(page.get('WIDTH')), float(page.get('HEIGHT'))
        
        output = []
        for line in root.findall('.//alto:TextLine', ns):
            for word in line.findall('alto:String', ns):
                content = word.get('CONTENT')
                x = int(float(word.get('HPOS')) / pw * 1000)
                y = int(float(word.get('VPOS')) / ph * 1000)
                output.append(f"<loc_{x}><loc_{y}>{content}")
        return " ".join(output)

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image (1024px high-res)
        image = Image.open(self.img_paths[idx]).convert('RGB').resize((1024, 1024))
        pixel_values = torch.tensor(list(image.getdata())).view(1024, 1024, 3).permute(2, 0, 1).float() / 255.0
        
        # Parse XML and Tokenize
        text = self.parse_alto(self.xml_paths[idx])
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        
        return pixel_values, tokens['input_ids'].squeeze()

# --- 3. TRAINING LOOP ---
def run_training(img_dir, xml_dir, batch_size=2, epochs=5, lr=5e-6):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = FullHTRSystem().cuda()
    
    dataset = ALTOParserDataset(img_dir, xml_dir, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f"Training system on {len(dataset)} images...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (pixel_values, input_ids) in enumerate(tqdm(loader)):
            pixel_values, input_ids = pixel_values.cuda(), input_ids.cuda()
            
            # Forward (Labels = input_ids for Causal LM)
            loss = model(pixel_values, input_ids, labels=input_ids)
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch} | Avg Loss: {epoch_loss/len(loader)}")
        torch.save(model.state_dict(), f'htr_full_model_ep{epoch}.pt')

if __name__ == '__main__':
    run_training(img_dir='./data/images', xml_dir='./data/alto_xml')
