import argparse
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from typing import Optional, List


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal DepthAI runner for a .blob model.")
    parser.add_argument(
        "--blob",
        type=Path,
        default=Path("nickv1-1_320_openvino_2022.1_6shave.blob"),
        help="Path to the OpenVINO blob",
    )
    parser.add_argument("--labels", type=Path, default=None, help="Path to labels txt (one per line)")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes if no labels file")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument(
        "--yolo-outputs",
        type=str,
        default="output1_yolov6r2,output2_yolov6r2,output3_yolov6r2",
        help="Comma-separated output layer names (defaults to all layers)",
    )
    parser.add_argument(
        "--apply-sigmoid-xywh",
        action="store_true",
        help="Apply sigmoid to x/y/w/h if your head outputs logits",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Camera FPS")
    parser.add_argument("--size", type=int, default=320, help="NN input width/height")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window rendering for more reliable performance measurements",
    )
    return parser.parse_args()


def build_pipeline(blob_path: Path, fps: float, size: int) -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(fps)
    cam.setPreviewSize(size, size)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(size, size)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    cam.preview.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(blob_path))
    manip.out.link(nn.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("nn")
    nn.out.link(xout.input)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    return pipeline


def load_labels(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        raise SystemExit(f"Labels file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def nms(boxes: List[np.ndarray], scores: List[float], iou_thresh: float) -> List[int]:
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while idxs:
        cur = idxs.pop(0)
        keep.append(cur)
        idxs = [i for i in idxs if iou_xyxy(boxes[cur], boxes[i]) < iou_thresh]
    return keep


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolo_v8_grid(
    outputs: dict,
    num_classes: int,
    conf_thresh: float,
    iou_thresh: float,
    apply_sigmoid_xywh: bool,
) -> List[tuple]:
    boxes: List[np.ndarray] = []
    scores: List[float] = []
    classes: List[int] = []

    for _, out in outputs.items():
        if out.ndim == 4 and out.shape[0] == 1:
            out = out[0]
        if out.ndim != 3:
            continue
        c, h, w = out.shape
        if c != 5 + num_classes:
            continue
        out = out.reshape(c, h * w).T  # (H*W, 5+C)

        xywh = out[:, 0:4]
        obj = out[:, 4]
        cls_scores = out[:, 5:]

        if apply_sigmoid_xywh:
            xywh = sigmoid(xywh)
        obj = sigmoid(obj)
        cls_scores = sigmoid(cls_scores)

        cls_id = np.argmax(cls_scores, axis=1)
        conf = obj * cls_scores[np.arange(cls_scores.shape[0]), cls_id]

        keep = conf >= conf_thresh
        if not np.any(keep):
            continue

        xywh = xywh[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]

        for i in range(xywh.shape[0]):
            x, y, w_box, h_box = xywh[i]
            x1 = x - w_box / 2.0
            y1 = y - h_box / 2.0
            x2 = x + w_box / 2.0
            y2 = y + h_box / 2.0
            boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            scores.append(float(conf[i]))
            classes.append(int(cls_id[i]))

    if not boxes:
        return []
    keep = nms(boxes, scores, iou_thresh)
    return [(boxes[i], scores[i], classes[i]) for i in keep]


def get_packet_timestamp_s(packet) -> float:
    for name in ("getTimestampDevice", "getTimestamp"):
        getter = getattr(packet, name, None)
        if getter is None:
            continue
        try:
            ts = getter()
        except Exception:
            continue
        if ts is None:
            continue
        if hasattr(ts, "total_seconds"):
            return float(ts.total_seconds())
        if hasattr(ts, "timestamp"):
            return float(ts.timestamp())
    return time.monotonic()


def nn_reader_worker(nn_q, lock, shared_state, stop_event):
    # Consume NN output asynchronously so UI/rendering does not throttle NN FPS measurements.
    while not stop_event.is_set():
        in_nn = nn_q.tryGet()
        if in_nn is None:
            time.sleep(0.001)
            continue

        layer_names = in_nn.getAllLayerNames()
        outputs = {}
        wanted_layers = shared_state["wanted_layers"] if shared_state["wanted_layers"] else layer_names
        for name in wanted_layers:
            if name in layer_names:
                data = np.array(in_nn.getLayerFp16(name), dtype=np.float32)
                outputs[name] = data

        packet_ts = get_packet_timestamp_s(in_nn)
        with lock:
            shared_state["latest_outputs"] = outputs
            shared_state["latest_packet_id"] += 1
            shared_state["nn_times"].append(packet_ts)


def main():
    args = parse_args()
    if not args.blob.exists():
        raise SystemExit(f"Blob not found: {args.blob}")

    labels = load_labels(args.labels)
    if not labels and args.num_classes is None:
        labels = ["car"]
    num_classes = args.num_classes or len(labels)
    if num_classes == 0:
        print("Warning: no labels/num-classes provided; detection parsing disabled.")

    pipeline = build_pipeline(args.blob, args.fps, args.size)

    with dai.Device(pipeline) as device:
        nn_q = device.getOutputQueue(name="nn", maxSize=30, blocking=False)
        rgb_q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        prev_time = 0.0
        fps_window = deque(maxlen=30)
        frame_count = 0
        conf_thresh = args.conf
        conf_changed = False

        wanted_layers = [s.strip() for s in args.yolo_outputs.split(",") if s.strip()] if args.yolo_outputs else []
        nn_lock = threading.Lock()
        shared_state = {
            "wanted_layers": wanted_layers,
            "latest_outputs": {},
            "latest_packet_id": -1,
            "nn_times": deque(maxlen=120),
        }
        stop_event = threading.Event()
        worker = threading.Thread(
            target=nn_reader_worker,
            args=(nn_q, nn_lock, shared_state, stop_event),
            daemon=True,
        )
        worker.start()

        detections: List[tuple] = []
        last_decoded_packet_id = -1
        last_print = 0.0

        window_name = "OAK-D YOLO"
        if not args.no_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, args.size, args.size)
            print("Running. Press 'q' to stop, '+'/'-' to adjust conf, 's' to save.")
        else:
            print("Running in no-display mode. Press Ctrl+C to stop.")
        try:
            while True:
                in_rgb = rgb_q.get()
                frame = in_rgb.getCvFrame()
                annotated = frame.copy() if not args.no_display else frame

                with nn_lock:
                    latest_outputs = shared_state["latest_outputs"]
                    latest_packet_id = shared_state["latest_packet_id"]
                    nn_times_list = list(shared_state["nn_times"])

                if num_classes > 0 and latest_outputs:
                    if latest_packet_id != last_decoded_packet_id or conf_changed:
                        detections = decode_yolo_v8_grid(
                            latest_outputs,
                            num_classes,
                            conf_thresh,
                            args.iou,
                            args.apply_sigmoid_xywh,
                        )
                        last_decoded_packet_id = latest_packet_id
                        conf_changed = False

                if not args.no_display:
                    h, w = annotated.shape[:2]
                    for box, score, cls_id in detections:
                        x1, y1, x2, y2 = box
                        # Expect normalized coordinates; scale to frame size.
                        x1 *= w
                        x2 *= w
                        y1 *= h
                        y2 *= h
                        x1 = int(max(0, min(w - 1, x1)))
                        y1 = int(max(0, min(h - 1, y1)))
                        x2 = int(max(0, min(w - 1, x2)))
                        y2 = int(max(0, min(h - 1, y2)))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = labels[cls_id] if cls_id < len(labels) else f"class{cls_id}"
                        cv2.putText(
                            annotated,
                            f"{label} {score:.2f}",
                            (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )

                current_time = time.time()
                if prev_time > 0:
                    fps_window.append(current_time - prev_time)
                fps = (len(fps_window) / sum(fps_window)) if fps_window else 0.0
                prev_time = current_time

                nn_fps = 0.0
                if len(nn_times_list) >= 2:
                    span = nn_times_list[-1] - nn_times_list[0]
                    if span > 0:
                        nn_fps = (len(nn_times_list) - 1) / span

                if not args.no_display:
                    cv2.putText(
                        annotated,
                        f"Display FPS: {fps:.1f}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"Infer FPS: {nn_fps:.1f}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"Conf: {conf_thresh:.2f}",
                        (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"Detections: {len(detections)}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("s"):
                        filename = f"detection_{frame_count}.jpg"
                        cv2.imwrite(filename, annotated)
                        print(f"Saved {filename}")
                        frame_count += 1
                    if key in (ord("+"), ord("=")):
                        conf_thresh = min(0.95, conf_thresh + 0.05)
                        conf_changed = True
                    if key in (ord("-"), ord("_")):
                        conf_thresh = max(0.05, conf_thresh - 0.05)
                        conf_changed = True
                else:
                    if current_time - last_print >= 1.0:
                        print(
                            f"Loop FPS: {fps:.1f} | Infer FPS: {nn_fps:.1f} | "
                            f"Conf: {conf_thresh:.2f} | Detections: {len(detections)}"
                        )
                        last_print = current_time
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            worker.join(timeout=1.0)
            if not args.no_display:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
