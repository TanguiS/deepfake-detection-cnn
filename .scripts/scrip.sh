SUBJECT_ROOT=/media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/subjects
SUBJECT_DF=/media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/subjects/dataframe.pkl

python main.py train --arch "Xception" \
-root $SUBJECT_ROOT \
-df $SUBJECT_DF \
-o "./log" \
-s 41 \
-d "85-10-5" \
--shape 256 \
-epoch 100 \
-b 64 \
-m "latest"

python main.py train --arch "EfficientNetB4" \
-root $SUBJECT_ROOT \
-df $SUBJECT_DF \
-o "./log" \
-s 41 \
-d "85-10-5" \
--shape 256 \
-epoch 100 \
-b 64 \
-m "latest"

python main.py train --arch "ResNet152V2" \
-root $SUBJECT_ROOT \
-df $SUBJECT_DF \
-o "./log" \
-s 41 \
-d "85-10-5" \
--shape 256 \
-epoch 100 \
-b 64 \
-m "latest"