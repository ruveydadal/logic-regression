# logic-regression
Machine Learning-1

Makine öğrenimi uygulamansının temel işlevselliğini ve kullanımını açıklar. Bu uygulama, bir veri kümesini kullanarak farklı sınıflandırma algoritmalarını nasıl uygulayacağınızı ve sonuçları nasıl değerlendireceğinizi gösterir.

## Başlangıç

Bu uygulamayı çalıştırmadan önce, aşağıdaki gereksinimlere dikkat edin:

- Python 3.x
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Kullanım

1. Verilerinizi `veriler.csv` adlı bir CSV dosyasından yükleyin.
2. Verileri özellikler (X) ve hedef değişken (y) olarak ayırın.
3. Verilerinizi eğitim ve test kümelerine ayırın.
4. Verileri ölçeklendirin (StandardScaler kullanarak).
5. Aşağıdaki sınıflandırma algoritmalarını uygulayın:
    - Logistic Regression
    - K-Nearest Neighborhood
    - Support Vector Machine
    - Gaussian Naive Bayes
    - Decision Tree Classifier
    - Random Forest Classifier

Her bir algoritmanın sonuçlarını incelemek için uygun satır kodunu kullanabilirsiniz.

## Parametre Ayarı

Uygulama içindeki bazı algoritma parametreleri örnekleme verinize ve probleminize bağlı olarak değiştirilebilir. Örneğin, `KNeighborsClassifier`'ın `n_neighbors` parametresini ayarlayabilirsiniz.

## Sonuçlar

Her algoritmanın sonuçları için bir karışıklık matrisi (confusion matrix) ve ROC eğrisi verilir. Bu sonuçları değerlendirmek için kodu kullanabilir ve algoritmaların performansını karşılaştırabilirsiniz.

## Notlar

- Bu kod, sadece eğitim amaçlıdır ve gerçek bir projenin gerektirdiği tüm özellikleri içermez.
- Veri setinizi ve algoritma seçimlerinizi projenize göre uyarlamak önemlidir.
- Bu README dosyasını ve kodu projeniz için özelleştirmeyi unutmayın.




