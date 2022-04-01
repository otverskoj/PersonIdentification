import cv2
import numpy as np
from src.person_detector import PersonDetector
from src.person_embedder import PersonEmbedder


def main(input_video_path: str,
         output_video_path: str) -> None:
    video_capture = cv2.VideoCapture(input_video_path)

    person_detector = PersonDetector()
    person_embedder = PersonEmbedder()
    # embedding_storage = EmbeddingStorage()

    # person_counter, frames_counter = 0, 0

    # out_resolution = 1920, 1080
    # writer = cv2.VideoWriter(output_video_path,
    #                          cv2.VideoWriter_fourcc(*'mp4v'),
    #                          25, 
    #                          out_resolution,
    #                          True)

    # while True:
    #     ret, frame = video_capture.read()
    #     if not ret:
    #         break
        
    #     resized_rgb_frame = cv2.cvtColor(
    #         cv2.resize(frame, out_resolution),
    #         cv2.COLOR_BGR2RGB
    #     )
    #     person_bboxes = person_detector.detect_persons(resized_rgb_frame)
    #     person_embeddings = person_embedder.embed(person_bboxes)

    #     # stored_person_embeddings = embedding_storage.person_embeddings()
    #     # similarity_matrix = LinalgUtils.count_similarity(stored_person_embeddings,
    #     #                                                  person_embeddings,
    #     #                                                  metric='cosine')
        
    #     # # ищем индексы строки значения, являющего максимумов в столбце
    #     # indices_to_swap = LinalgUtils.swap_indices(similarity_matrix, 
    #     #                                            metric='cosine')
    #     # embedding_storage.update_embeddings(person_embeddings, indices_to_swap)
        
    #     # # индексы строк новых для нас людей
    #     # indices_to_add = np.array(
    #     #     filter(
    #     #         lambda i: i not in indices_to_swap, 
    #     #         range(person_embeddings.shape[1])
    #     # ))
    #     # embedding_storage.add_embeddings(person_embeddings)

    #     # person_counter += indices_to_add.shape[0]

    #     # data = tuple(
    #     #     {
    #     #         'type': 'text',
    #     #         'coords': (),
    #     #         'content': f'Total persons: {person_counter}'
    #     #     }
    #     # )
    #     # frame_to_write = DrawUtils.draw(data, resized_rgb_frame[:])

    #     frame_to_write = cv2.cvtColor(resized_rgb_frame,
    #                                   cv2.COLOR_RGB2BGR)
    #     for bbox in person_bboxes:
    #         x, y, w, h = list(map(int, bbox))
    #         frame_to_write = cv2.rectangle(frame_to_write, 
    #                                        (x, y), (x + w, y + h), 
    #                                        (0, 255, 0), 2)
    #     if frames_counter % 125 == 0:
    #         print(frames_counter)
    #     writer.write(frame_to_write)
    #     frames_counter += 1


if __name__ == '__main__':
    main(r'data\input\video\clip.mp4', r'data\output\video\clip.mp4')
