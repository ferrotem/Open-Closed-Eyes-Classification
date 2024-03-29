# Open and Closed Eyes Classification
Для выполнения задания были сделаны следующие шаги:

1. Сперва я проверил общеизвестные алгоритмы кластеризации для разметки данных. Но оно не дало желаемых результатов.
2. Для того что бы решить проблему не размеченных данных использовал следующюю методику: One Shot Learning with Siamese Networks. Это методика используется когда размеченных данных мало. Для этого я выбрал 50 фотографий с закрытыми глазами и 50 фотографий с открытыми глазами. 
3. Два параллельных Siamese networks обучаются с функцией Contrastive loss,  где при одинаковых классов, loss будет меньше, а для разных он больше. Основа алгоритма может быть найдена [5,4]. 
4. Во время Inference в один из параллельных нетворков мы вводим выбранные нами открытые глаза(20 фото), а во второй нетворк тестовое фото(20 копий одного и того же фото), как только средняя схожесть будет больше 0.5, даем лейбл 1.

Notes

Есть не решенная проблема Exploding gradients.

Reference
1. https://github.com/keras-team/keras-io/blob/master/examples/vision/semisupervised_simclr.py
2. https://arxiv.org/pdf/2011.10566.pdf
3. https://keras.io/examples/vision/simsiam/
4. http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf
5. https://papers.nips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf
