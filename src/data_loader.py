import json
from pathlib import Path
from PIL import Image


class SROIEDataLoader:
    """
    Loads images and labels from the SROIE dataset.
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}")
        
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        print(f"Data loader ready → {self.data_dir}")

    # Helper function
    def _get_base_dir(self, split):
        """Returns train or test directory based on split"""
        return self.train_dir if split == "train" else self.test_dir
    # List receipt IDs
    def get_sample_ids(self, split="train"):
        """Get list of all receipt IDs in the dataset"""
        img_dir = self._get_base_dir(split) / "img"
        if not img_dir.exists():
            raise FileNotFoundError(f"Image folder missing: {img_dir}")
        return [img.stem for img in img_dir.glob("*.jpg")]

    # Load receipt image
    def load_image(self, sample_id, split="train"):
        """Load a single receipt image"""
        img_path = self._get_base_dir(split) / "img" / f"{sample_id}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path)

    # Load text boxes (ground truth)
    def load_ground_truth_boxes(self, sample_id, split="train"):
        """Load ground truth text and coordinates"""
        box_path = self._get_base_dir(split) / "box" / f"{sample_id}.txt"
        if not box_path.exists():
            return []
        
        boxes = []
        with open(box_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                
                coords = list(map(int, parts[:8]))
                text = ",".join(parts[8:]).strip()
                boxes.append({
                    "coords": coords,
                    "text": text
                })
        return boxes

    # Load structured entities
    def load_structured_data(self, sample_id, split="train"):
        """Load key-value pairs (company, date, total, address)"""
        entity_path = self._get_base_dir(split) / "entities" / f"{sample_id}.txt"
        if not entity_path.exists():
            return {}
        
        with open(entity_path, encoding="utf-8") as f:
            return json.load(f)

    # Load everything together
    def load_complete_sample(self, sample_id, split="train"):
        """Load image, boxes, and entities for one receipt"""
        return {
            "sample_id": sample_id,
            "image": self.load_image(sample_id, split),
            "boxes": self.load_ground_truth_boxes(sample_id, split),
            "entities": self.load_structured_data(sample_id, split)
        }

# Test code - run this file directly to verify it works
if __name__ == "__main__":
    print("Testing Data Loader")
    
    # Initialize
    loader = SROIEDataLoader("data/raw/SROIE2019")
    
    # Get sample list
    samples = loader.get_sample_ids("train")
    print(f" Found {len(samples)} training samples")
    
    # Load first sample
    if samples:
        sample = loader.load_complete_sample(samples[0])
        print(f"Loaded: {sample['sample_id']}")
        print(f"   Image: {sample['image'].size}")
        print(f"   Boxes: {len(sample['boxes'])}")
        print(f"   Entities: {list(sample['entities'].keys())}")
        print("\n Test passed!")
    
    print("=" * 60)