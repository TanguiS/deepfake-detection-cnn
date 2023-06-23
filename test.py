from pathlib import Path
from bench.Evaluation import ModelEvaluator, decode_prediction
from bench.extract.face.FaceDetectorResult import load_face_detection_model

model_path = Path('C:\\WORK\\deepfake-detection-cnn\\log\\VGG16_deepfake-detection_02.h5')
img_path_fake = Path('D:\\storage-photos\\benchmark\\p384dfudt\\1000\\00090.png')
img_path_real = Path('D:\\storage-photos\\benchmark\\reference_face.png')

yunet_model = Path('C:\\WORK\\model')

shape = (128, 128)

face_detector = load_face_detection_model(yunet_model, input_size=shape)

print(f"Loading models {model_path.stem} ...")
evaluator = ModelEvaluator(model_path, shape)
print("Model loaded.")

print("Fast evaluation on a fake image : ")
predict = evaluator.evaluate(img_path_fake, face_detector)
test = decode_prediction(predict)
print(f"Prediction : {predict}")
print(f"Decoded prediction : {test}")

print("Fast evaluation on a real image : ")
predict = evaluator.evaluate(img_path_real, face_detector)
test = decode_prediction(predict)
print(f"Prediction : {predict}")
print(f"Decoded prediction : {test}")
