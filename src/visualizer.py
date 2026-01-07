from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


class ConfidenceVisualizer:
    """
    Draws colored boxes on images to highlight confidence levels.
    
    Color scheme:
    - Green: High confidence (reliable)
    - Yellow: Medium confidence (check if important)
    - Red: Low confidence (needs review)
    """
    
    def __init__(self, low_threshold=60, high_threshold=80):
        """
        Initialize visualizer
        
        Args:
            low_threshold: Below this = red boxes
            high_threshold: Above this = green boxes
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # Color scheme
        self.colors = {
            'high': (0, 255, 0),      # Green
            'medium': (255, 255, 0),  # Yellow
            'low': (255, 0, 0)        # Red
        }
        
        print(f"Visualizer ready (color-coded by confidence)")
    
    # Main visualization function

    def visualize_confidence(self, image, ocr_result, save_path=None):
        """
        Draw colored boxes on image based on confidence
        
        Args:
            image: PIL Image object
            ocr_result: Output from ConfidenceOCR
            save_path: Optional path to save annotated image
        
        Returns:
            Annotated PIL Image
        """
        # Create a copy to draw on
        img_annotated = image.copy()
        draw = ImageDraw.Draw(img_annotated)
        
        # Draw boxes for each word
        for word in ocr_result['words']:
            conf = word['confidence']
            bbox = word['bbox']
            
            # Determine color and thickness based on confidence
            if conf >= self.high_threshold:
                color = self.colors['high']
                width = 1
            elif conf >= self.low_threshold:
                color = self.colors['medium']
                width = 2
            else:
                color = self.colors['low']
                width = 3
            
            # Draw rectangle
            draw.rectangle(bbox, outline=color, width=width)
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img_annotated.save(save_path)
            print(f" Saved annotated image → {save_path}")
        
        return img_annotated

    # Highlight only low confidence
    def highlight_review_areas(self, image, analysis, save_path=None):
        """
        Highlight ONLY low confidence words (for focused review)
        
        Args:
            image: PIL Image object
            analysis: Output from ConfidenceAnalyzer
            save_path: Optional path to save
        
        Returns:
            Annotated PIL Image with only red boxes
        """
        img_annotated = image.copy()
        draw = ImageDraw.Draw(img_annotated)
        
        # Draw only low confidence words
        low_words = analysis['low_confidence']['words']
        for word in low_words:
            bbox = word['bbox']
            draw.rectangle(bbox, outline=self.colors['low'], width=4)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img_annotated.save(save_path)
            print(f" Saved review map → {save_path}")
        
        return img_annotated
    
    # Create side-by-side comparison
    def create_comparison(self, original, annotated, save_path=None):
        """
        Create side-by-side comparison (original | annotated)
        
        Args:
            original: Original PIL Image
            annotated: Annotated PIL Image
            save_path: Optional path to save
        
        Returns:
            Combined PIL Image
        """
        # Calculate new dimensions
        width = original.width * 2
        height = original.height
        
        # Create blank canvas
        combined = Image.new('RGB', (width, height), 'white')
        
        # Paste images side by side
        combined.paste(original, (0, 0))
        combined.paste(annotated, (original.width, 0))
        
        # Add divider line
        draw = ImageDraw.Draw(combined)
        draw.line([(original.width, 0), (original.width, height)], fill='black', width=3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            combined.save(save_path)
            print(f" Saved comparison → {save_path}")
        
        return combined
    

    # Generate legend
    def add_legend(self, image):
        """
        Add color legend to image
        
        Args:
            image: PIL Image
        
        Returns:
            Image with legend
        """
        img_with_legend = image.copy()
        draw = ImageDraw.Draw(img_with_legend)
        
        # Legend position (top-left corner)
        x, y = 10, 10
        box_size = 20
        gap = 5
        
        # Draw legend boxes
        legends = [
            ('high', 'High confidence (≥80%)'),
            ('medium', 'Medium (60-79%)'),
            ('low', 'Low (<60%) - Review!')
        ]
        
        for conf_type, label in legends:
            color = self.colors[conf_type]
            # Draw colored box
            draw.rectangle([x, y, x + box_size, y + box_size], 
                          outline=color, width=3)
            y += box_size + gap
        
        return img_with_legend

# Test code
if __name__ == "__main__":
    from data_loader import SROIEDataLoader
    from ocr_engine import ConfidenceOCR
    from confidence_scorer import ConfidenceAnalyzer
    
 
    print(" Testing Visualizer")
    
    # Load sample
    loader = SROIEDataLoader("data/raw/SROIE2019")
    samples = loader.get_sample_ids("train")
    sample = loader.load_complete_sample(samples[0])
    
    print(f" Testing on: {sample['sample_id']}")
    
    # Run OCR
    ocr = ConfidenceOCR()
    ocr_result = ocr.extract_text_with_confidence(sample['image'])
    
    # Analyze
    analyzer = ConfidenceAnalyzer()
    analysis = analyzer.analyze(ocr_result)
    
    # Visualize
    visualizer = ConfidenceVisualizer()
    
    # Create output directory
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print()
    print(" Creating visualizations...")
    
    # 1. Full confidence map
    confidence_map = visualizer.visualize_confidence(
        sample['image'],
        ocr_result,
        save_path=output_dir / f"{sample['sample_id']}_confidence_map.jpg"
    )
    
    # 2. Review areas only
    review_map = visualizer.highlight_review_areas(
        sample['image'],
        analysis,
        save_path=output_dir / f"{sample['sample_id']}_review_map.jpg"
    )
    
    # 3. Side-by-side comparison
    comparison = visualizer.create_comparison(
        sample['image'],
        confidence_map,
        save_path=output_dir / f"{sample['sample_id']}_comparison.jpg"
    )
    
    print()
    print(" Visualizations created!")
    print(f" Check folder: {output_dir.absolute()}")
    print()
    
    # Display one
    print(" Opening confidence map...")
    try:
        confidence_map.show()
    except:
        print("  Can't auto-display. Check the saved files manually.")
    
    print()
    print(" Test passed!")
    