# GAN   
   
![generative_model](https://user-images.githubusercontent.com/59756209/74510163-7f48bd80-4f46-11ea-9ca0-f3a602b273a5.PNG)   
   
어떤 값을 분류하거나 예측하는 모델인 ' Discriminative Model(구분 모델)'과 달리, GAN은  'Generative Model(생성 모델)' 이다.   
생성 모델은 주어진 train data feautre를 학습하여 train data와 유사한 new data를 generate하는 모델이다.   
즉, train data의 distribution을 학습하여 비슷한 distribution을 나타내는 data sampling을 통해 새로 generate하는 것이다.   
   
## 1. Latent Variable(잠재번수)   
잠재변수는 숨겨진 variable으로써 data에 직접적으로 나타나지 않지만 현재 data distribution을 만드는데 영향을 끼치는 variable이다.   
따라서, 어떤 data의 LV를 알아내면 LV를 이용해서 해당 data와 유사한 data를 generate할 수 있다.   
즉, LV는 data의 형태를 결정하는 특징으로 생각할 수 있다.   
예를 들어, 우리가 학습하고 generate하고자 하는 data가 사람 얼굴 image라면 적절한 LV는 사람의 성별이 될 수 있습니다.   
LV를 사람의 성별로 간주할 경우, 이 사람이 남자인지 여자인지를 나타내는 1차원 Boolean feature value만을 가지고도 generate하고자 하는 data의 형태를 어느정도 결정할 수 있다.   
이에 더해서, 사람의 표정, 촬영한 카메라의 각도 등이 적절한 VL이 될 수 있다.   
여기서는, MNIST 필기체 data이므로 필기획의 기울기, image를 나타내는 lable등이 적절한 LV라고 될 수 있다.   
   
## 2. GAN   
GAN은 generator와 discriminator라는 2가지 부분으로 구성되어 있다.   
GAN의 개념을 직관적으로 이해하기 위해서 많이 사용하는 예시인 경찰(discriminator)와 위조지폐 생성범(generator)이다.   
위조 지폐 생성범(generator)는 경찰(discriminator)를 속이기 위해서 최대한 진짜 지폐와 구분이 되지않는 위조지폐를 생성하려고 노력한다.   
이에 반해 경찰은 위조지폐 생성범이 생성한 위조지폐와 진짜 지폐를 최대한 정확하게 구분할 수 있도록 노력한다.  
경찰과 위조지폐 생성범이 서로 노력해서 계속 학습을 진행하면 경찰이 위조지폐 생성범이 생성한 위조지폐와 진짜 지폐를 50% 확률로 구분할 수 있게 되는 균형점에서 학습이 종료한다.   
결과적으로, generator는 data와 유사한 data distribution 학습을 하게된다.
