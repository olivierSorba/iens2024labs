# Testing VisionAgent (<https://va.landing.ai/agent>)

## Source image

![source image](./images/seance-engagement-toits-paris-sarahcouturier-lamarieeauxpiedsnus-19.jpgseance-engagement-toits-paris-sarahcouturier-lamarieeauxpiedsnus-19.jpg)

## Painting

### `paint all the tin roofs in cyan, paint the river in red`

- Answer: ![paint_result](images/paint_result.png)
- Answer: ![paint_result](images/Florence-2Sam2Image%20output.png)


```text
Executing code.

The image has been successfully edited. Tin roofs have been painted cyan, and the river has been painted red. The edited image is saved at /home/user/edited_roofs_and_river.jpg. Would you like to view the edited image or do you have any other tasks?
```

- generated code

    <details>
    <summary>Click to expand code</summary>

    ```python  
    import os  
    import numpy as np  
    from vision_agent.tools import *  
    from typing import *  
    from pillow_heif import register_heif_opener  
    register_heif_opener()  
    import vision_agent as va  
    from vision_agent.tools import register_tool  
    
    
    import numpy as np  
    from vision_agent.tools import load_image, florence2_sam2_image, save_image  
    
    def edit_roofs_and_river(image_path):  
        # Load the image  
        image = load_image(image_path)  
        
        # Use florence2_sam2_image to segment tin roofs and river  
        segmentation_results = florence2_sam2_image("tin roofs, river", image)  
        
        # Create a copy of the original image for editing  
        edited_image = image.copy()  
        
        # Define colors (BGR format for OpenCV)  
        cyan_color = np.array([255, 255, 0])  # Cyan in BGR  
        red_color = np.array([0, 0, 255])     # Red in BGR  
        
        # Process each segmented object  
        for obj in segmentation_results:  
            mask = obj['mask']  
            label = obj['label'].lower()  
            
            if 'roof' in label:  
                # Apply cyan color to tin roofs  
                edited_image[mask == 1] = cyan_color  
            elif 'river' in label:  
                # Apply red color to river  
                edited_image[mask == 1] = red_color  
        
        # Save the edited image  
        output_path = '/home/user/edited_roofs_and_river.jpg'  
        save_image(edited_image, output_path)  
        
        return output_path

    # The function can be called like this:
    # result = edit_roofs_and_river('/home/user/GopYcha_seance-engagement-toits-paris-sarahcouturier-lamarieeauxpiedsnus-19.jpg')
    ```

    </details>

- Tool used:

    ```text
    Florence-2 Sam2 Image
    A tool that can segment multiple objects given a text prompt such as category names or referring expressions. The categories in the text prompt are separated by commas. It returns a list of bounding boxes, label names, mask file names and associated probability scores of 1.0.
    ```

## Distance Calculation

### `what is the distance between the chimney on the left and cathedral ?`

- Answer

    ```text
    Based on the analysis of the image, the distance between the chimney on the left and the cathedral is approximately ⚠️ 349.18 pixels ⚠️. The chimney is located in the lower left part of the image, with its bounding box coordinates being [0.1, 0.72, 0.13, 0.78]. The cathedral is situated more towards the center of the image, with its bounding box coordinates being [0.37, 0.33, 0.51, 0.56]. A visualization of these detections has been saved as '/home/user/distance_visualization.jpg'. Would you like to view this visualization or do you need any further information about the results?
    ```

- tool:

    ```text
    Florence2 Phrase Grounding
    A tool that can detect multiple objects given a text prompt which can be object names or caption. You can optionally separate the object names in the text with commas. It returns a list of bounding boxes with normalized coordinates, label names and associated probability scores of 1.0.
    ```

- generated code:

    <details>
    <summary>Click to expand code</summary>

    ```python
    import os
    import numpy as np
    from vision_agent.tools import *
    from typing import *
    from pillow_heif import register_heif_opener
    register_heif_opener()
    import vision_agent as va
    from vision_agent.tools import register_tool


    from vision_agent.tools import load_image, florence2_phrase_grounding, closest_box_distance, overlay_bounding_boxes

    def detect_and_measure_distance(image_path):  
        # Load the image  
        image = load_image(image_path)  
        
        # Detect the chimney and cathedral  
        detections = florence2_phrase_grounding("chimney, cathedral", image)  
        
        # Extract bounding boxes  
        chimney = None  
        cathedral = None  
        for detection in detections:  
            if detection['label'] == 'chimney':  
                chimney = detection['bbox']  
            elif detection['label'] == 'cathedral':  
                cathedral = detection['bbox']  
        
        # Check if both objects were detected  
        if not chimney or not cathedral:  
            return {  
                "error": "Could not detect both chimney and cathedral.",  
                "chimney_bbox": chimney,  
                "cathedral_bbox": cathedral,  
                "visualized_image": overlay_bounding_boxes(image, detections)  
            }  
        
        # Get image dimensions  
        height, width = image.shape[:2]  
        
        # Calculate distance  
        distance = closest_box_distance(chimney, cathedral, (height, width))  
        
        # Visualize the results  
        visualized_image = overlay_bounding_boxes(image, detections)  
        
        return {  
            "distance": distance,  
            "chimney_bbox": chimney,  
            "cathedral_bbox": cathedral,  
            "visualized_image": visualized_image  
        }

    # The function can be called like this:
    # result = detect_and_measure_distance("/home/user/GopYcha_seance-engagement-toits-paris-sarahcouturier-lamarieeauxpiedsnus-19.jpg")
    ```

- result :

    ```json
    [{'distance': 349.17760523836574,
        'chimney_bbox': [0.1, 0.72, 0.13, 0.78],
        'cathedral_bbox': [0.37, 0.33, 0.52, 0.56],
        'visualized_image': array([[[205, 214, 213],...,[ 81,  94,  87]]], dtype=uint8)
        }
    ]
    ```

- generated code

    <details>
    <summary>Click to expand code</summary>

    ```python
    import os
    import numpy as np
    from vision_agent.tools import *
    from typing import *
    from pillow_heif import register_heif_opener
    register_heif_opener()
    import vision_agent as va
    from vision_agent.tools import register_tool


    from vision_agent.tools import load_image, florence2_phrase_grounding, closest_box_distance, overlay_bounding_boxes

    def detect_and_measure_distance(image_path):
        # Load the image
        image = load_image(image_path)
        
        # Detect the chimney and cathedral
        detections = florence2_phrase_grounding("chimney, cathedral", image)
        
        # Extract bounding boxes
        chimney = None
        cathedral = None
        for detection in detections:
            if detection['label'] == 'chimney':
                chimney = detection['bbox']
            elif detection['label'] == 'cathedral':
                cathedral = detection['bbox']
        
        # Check if both objects were detected
        if not chimney or not cathedral:
            return {
                "error": "Could not detect both chimney and cathedral.",
                "chimney_bbox": chimney,
                "cathedral_bbox": cathedral,
                "visualized_image": overlay_bounding_boxes(image, detections)
            }
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate distance
        distance = closest_box_distance(chimney, cathedral, (height, width))
        
        # Visualize the results
        visualized_image = overlay_bounding_boxes(image, detections)
        
        return {
            "distance": distance,
            "chimney_bbox": chimney,
            "cathedral_bbox": cathedral,
            "visualized_image": visualized_image
        }

        # The function can be called like this:
        # result = detect_and_measure_distance("/home/user/GopYcha_seance-engagement-toits-paris-sarahcouturier-lamarieeauxpiedsnus-19.jpg")
    ```

    </details>

## Perspective distance

### `what is the geographic distance between the chimney on the left and cathedral, in meters ? Take perspective into account.`

- Answer

```text
Executing code.

Based on the analysis of the image and taking perspective into account, the estimated geographic distance between the chimney on the left and the cathedral is ⚠️approximately 246.97 meters⚠️. This calculation considers the depth information in the image, which helps account for the perspective and provides a more accurate estimate of the real-world distance. Would you like any further information or analysis of this result?
```

- generated code

    <details>
    <summary>Click to expand code</summary>

    ```python
    import os
    import numpy as np
    from vision_agent.tools import *
    from typing import *
    from pillow_heif import register_heif_opener
    register_heif_opener()
    import vision_agent as va
    from vision_agent.tools import register_tool


    from vision_agent.tools import load_image, owl_v2_image, depth_anything_v2
    import numpy as np

    def calculate_distance(image_path):
        # Load the image
        image = load_image(image_path)
        height, width = image.shape[:2]

        # Detect chimney and cathedral
        detections = owl_v2_image("chimney, cathedral", image, box_threshold=0.1)
        
        chimney = None
        cathedral = None
        for detection in detections:
            if detection['label'] == 'chimney' and detection['bbox'][0] < 0.5:  # Left side of image
                chimney = detection
            elif detection['label'] == 'cathedral':
                cathedral = detection
        
        if not chimney or not cathedral:
            return "Could not detect both chimney and cathedral"

        # Generate depth map
        depth_map = depth_anything_v2(image)

        # Function to get average depth from a bounding box
        def get_avg_depth(bbox):
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [width, height, width, height])]
            return np.mean(depth_map[y1:y2, x1:x2])
        
        # Estimate focal length (assuming a 60-degree field of view)
        focal_length = width / (2 * np.tan(np.radians(30)))

        def get_world_coords(bbox, depth):
            cx, cy = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            x = (cx - 0.5) * depth * width / focal_length
            y = (cy - 0.5) * depth * height / focal_length
            return x, y, depth

        # Get world coordinates for chimney and cathedral
        chimney_depth = get_avg_depth(chimney['bbox'])
        cathedral_depth = get_avg_depth(cathedral['bbox'])
        chimney_coords = get_world_coords(chimney['bbox'], chimney_depth)
        cathedral_coords = get_world_coords(cathedral['bbox'], cathedral_depth)

        # Calculate Euclidean distance
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(chimney_coords, cathedral_coords)))

        # Convert to meters (assuming depth values are in meters)
        return distance
        ```

    </details>

- result : `[131.3098753298001]` (note the agent did the feet to meters conversion)
