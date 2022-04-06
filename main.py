import cv2
import numpy as np
from src.person_detector import PersonDetector
from src.person_embedder import PersonEmbedder
from src.embedding_storage import EmbeddingStorage


def main(input_video_path: str,
         output_video_path: str) -> None:
    video_capture = cv2.VideoCapture(input_video_path)

    person_detector = PersonDetector()
    person_embedder = PersonEmbedder()
    embedding_storage = EmbeddingStorage()

    person_counter, frames_counter = 0, 0

    out_resolution = 2560, 1440
    writer = cv2.VideoWriter(output_video_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             25, 
                             out_resolution,
                             True)

    while True:
        ret, frame = video_capture.read()
        if not ret or frames_counter == 500:
            break
        
        person_bboxes = person_detector.detect_persons(frame)
        person_embeddings = person_embedder.embed_persons(frame, person_bboxes)

        if len(embedding_storage) != 0:
            _, indices = embedding_storage.filtered_search(person_embeddings, k=2, threshold=0.9)
            embedding_storage.remove(indices[:, 0])
            person_counter += person_embeddings.shape[0] - indices.shape[0]
        
        if len(embedding_storage) == 0:
            person_counter += person_embeddings.shape[0]
        
        embedding_storage.add(person_embeddings)
                
        frame_to_write = cv2.resize(frame, out_resolution)
        for bbox in person_bboxes:
            frame_to_write = cv2.rectangle(frame,
                                           bbox[:2], bbox[2:],
                                           (0, 255, 0), 2)
            frame_to_write = cv2.putText(frame_to_write,
                                         f'Total persons: {person_counter}',
                                         (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         1, (0, 255, 0), 5)
        if frames_counter % 125 == 0:
            print(frames_counter)
        writer.write(frame_to_write)
        frames_counter += 1


if __name__ == '__main__':
    main(r'data\input\video\clip.mp4', r'data\output\video\clip.mp4')
