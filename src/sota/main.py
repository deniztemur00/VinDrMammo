import torch

from retinanet import model
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sota import SOTA
from GMIC_adaptation.config import GlobalConfig
import matplotlib.pyplot as plt


def main():
    # --- Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Model and input parameters
    batch_size = 2
    num_classes = 5
    img_size = 512

    retinanet = model.resnet18(num_classes=num_classes, pretrained=True)
    # Instantiate the model
    # Create dummy input data
    dummy_img_batch = torch.rand(batch_size, 3, img_size, img_size).to(device)
    # Dummy annotations: list of tensors, one for each image in the batch
    # Each tensor is [num_boxes, 5] -> [x1, y1, x2, y2, class_id]
    dummy_annotations = [torch.rand(5, 5).to(device) for _ in range(batch_size)]
    dummy_annotations[0][:, 4] = torch.randint(
        0, num_classes, (5,)
    )  # Ensure valid class IDs
    dummy_annotations[1][:, 4] = torch.randint(0, num_classes, (5,))
    dummy_annotations = torch.stack(dummy_annotations)

    # --- Training Mode Test ---
    print("\n" + "=" * 20 + " TESTING IN TRAINING MODE " + "=" * 20)
    retinanet.train()

    # Perform forward pass
    loss, rois = retinanet([dummy_img_batch, dummy_annotations])

    print(f"Loss returned: {loss}")
    print(f"Returned {len(rois)} sets of ROIs (one per image in batch).")

    # Inspect the ROIs for the first image in the batch
    if len(rois) > 0:
        print(rois)
        scores, labels, boxes = rois[0]
        print("\n--- ROIs for first image in batch ---")
        print(f"Number of ROIs found: {boxes.shape[0]}")
        if boxes.shape[0] > 0:
            print(f"Scores shape: {scores.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Boxes shape: {boxes.shape}")
            print("Example Box (first one):")
            print(f"  Score: {scores[0]:.4f}, Label: {labels[0]}, Box: {boxes[0]}")

    # --- Evaluation Mode Test ---
    print("\n" + "=" * 20 + " TESTING IN EVALUATION MODE " + "=" * 20)
    retinanet.eval()

    # No need for annotations in eval mode
    with torch.no_grad():
        final_detections = retinanet(dummy_img_batch)

    print(
        f"Returned {len(final_detections)} sets of detections (one per image in batch)."
    )

    # Inspect the detections for the first image in the batch
    if len(final_detections) > 0:
        scores, labels, boxes = final_detections[0]
        print("\n--- Detections for first image in batch ---")
        print(f"Number of detections found: {boxes.shape[0]}")
        if boxes.shape[0] > 0:
            print(f"Scores shape: {scores.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Boxes shape: {boxes.shape}")
            print("Example Detection (first one):")
            print(f"  Score: {scores[0]:.4f}, Label: {labels[0]}, Box: {boxes[0]}")


def main2():
    # --- Setup ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Model and input parameters
    batch_size = 5
    img_size = 512

    config = GlobalConfig(device_type=device, crop_shape=(128, 128))
    sota_model = SOTA(config=config).to(device)

    # Create dummy input data
    dummy_img_batch = torch.rand(batch_size, 3, img_size, img_size).to(device)

    # Dummy annotations: list of tensors, one for each image in the batch
    # Each tensor is [num_boxes, 5] -> [x1, y1, x2, y2, class_id]
    dummy_annotations = [torch.rand(10, 5).to(device) * 10 for _ in range(batch_size)]
    for annotations in dummy_annotations:
        annotations[:, 4] = torch.randint(
            0, config.n_findings, (10,)
        )  # Ensure valid class IDs
    # dummy_annotations = torch.stack(dummy_annotations)

    #    sota_model.train()  # Set model to training mode
    #    print("\n" + "=" * 20 + " TESTING SOTA MODEL IN TRAINING MODE " + "=" * 20)
    #
    #    # Perform forward pass
    #    loss_dict = sota_model(dummy_img_batch, dummy_annotations)
    #    #print(f"detection Loss returned: {detection_loss}")
    #    #print(f"birads logits: {fusion_birads}")
    #    #print(f"density logits: {density_logits}")
    #    print(f"Loss returned: {loss_dict}")

    print("\n" + "=" * 20 + " TESTING SOTA MODEL IN EVALUATION MODE " + "=" * 20)
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    sota_model.eval()
    mAP = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    with torch.no_grad():
        inference_results = sota_model(dummy_img_batch)

    preds, targets = _update_map(inference_results, dummy_annotations)

    # print(f"Targets: {_targets}")
    mAP.update(preds, targets)
    fig, ax = mAP.plot()
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Save the modified figure, ensuring the legend is not cut off
    fig.savefig("mAP_plot.png", bbox_inches='tight')

    results = mAP.compute()
    print(f"mAP results: {results}")


def _update_map(outputs, dummy_annotations):
    """
    Processes model outputs and ground truth annotations for a full batch
    to prepare them for the mAP metric.
    """
    all_preds = []
    all_targets = []

    # outputs['detections'] is a list of (scores, labels, boxes) tuples, one for each image.
    # dummy_annotations is a list of annotation tensors, one for each image.
    batch_size = len(outputs["detections"])

    for i in range(batch_size):
        # --- Process Predictions for image i ---
        cls_scores, cls_indices, bboxes = outputs["detections"][i]

        # Move tensors to CPU
        cls_scores = cls_scores.cpu()
        cls_indices = cls_indices.cpu()
        bboxes = bboxes.cpu()

        # Filter out invalid boxes where x1 > x2 or y1 > y2
        valid_indices = (bboxes[:, 0] < bboxes[:, 2]) & (bboxes[:, 1] < bboxes[:, 3])

        # Append the processed predictions for the current image to the list
        all_preds.append(
            {
                "boxes": bboxes[valid_indices],
                "labels": cls_indices[valid_indices],
                "scores": cls_scores[valid_indices],
            }
        )

        # --- Process Targets for image i ---
        annotation = dummy_annotations[i].cpu()

        # Append the processed targets for the current image to the list
        all_targets.append(
            {
                "boxes": annotation[:, :4],
                "labels": annotation[:, 4].long(),
            }
        )
        print(f"Processed image {i + 1}/{batch_size}:")

    return all_preds, all_targets


# print(f"detections: {detections}")
if __name__ == "__main__":
    main2()
