
# Laporan Proyek Machine Learning - Evry Nazyli Ciptanto

## Domain Proyek

Tingkat pergantian karyawan (*employee turnover*) menjadi masalah yang signifikan bagi banyak perusahaan. Menurut data dari Gallup, meskipun ada penurunan persentase pekerja yang kurangnya komitmen terhadap pekerjaa, yaitu dari 18% pada 2022 menjadi 16% pada 2023, masalah keterlibatan karyawan tetap menjadi isu besar. Pada 2023, sekitar 50% karyawan di seluruh dunia tercatat tidak terlibat secara aktif dalam pekerjaan mereka, yang sering kali disebut sebagai *quiet quitting* atau pengunduran diri diam diam [[1]](https://www.gallup.com/workplace/608675/new-workplace-employee-engagement-stagnates.aspx).

![turnover](https://github.com/user-attachments/assets/198a4117-49c0-4cf8-aca6-03bfa0ce611d)


Dalam data tersebut terlihat *Turnover Rate (%)* dalam rentang tahun 2020 - 2023 di berbagai sektor [[2]](https://www.award.co/blog/employee-turnover-rates). 
Untuk itu, perusahaan perlu memahami faktor-faktor yang menyebabkan karyawan meninggalkan perusahaan dan bagaimana cara mengurangi risiko ini. Banyak faktor yang dapat memengaruhi keputusan seorang karyawan untuk bertahan atau keluar, seperti tingkat kepuasan kerja, kesempatan untuk berkembang, gaji, dan keseimbangan antara pekerjaan dan kehidupan pribadi.

Dengan menggunakan data historis karyawan seperti umur, pengalaman kerja, kepuasan, gaji, pendidikan dan faktor-faktor lain perusahaan dapat mengembangkan model prediksi untuk mengidentifikasi karyawan yang berisiko tinggi untuk meninggalkan perusahaan. Pendekatan ini memungkinkan tim HR untuk mengambil langkah-langkah preventif yang lebih tepat sasaran dan meningkatkan retensi karyawan.

Melalui analisis data dan machine learning, proyek ini bertujuan untuk memprediksi kemungkinan karyawan akan mengundurkan diri atau tidak, sehingga perusahaan dapat lebih proaktif dalam merancang kebijakan yang mendukung karyawan dan mengurangi tingkat pergantian karyawan.

## Business Understanding

### Problem Statements

Dari latar belakang yang telah dijelaskan di atas, diperoleh rumusan masalah yang akan diatasi dalam proyek ini, yaitu:

 1. Di antara berbagai fitur yang tersedia, fitur mana yang memiliki dampak terbesar terhadap potensi karyawan meninggalkan perusahaan?  
 2. Apa model terbaik yang dapat digunakan untuk memprediksi karyawan meninggalkan perusahaan?

### Goals

Berdasarkan *problem statements* yang telah dijelaskan sebelumnya, tujuan dari proyek ini adalah:

1. Menentukan fitur-fitur yang paling berpengaruh terhadap potensi kemungkinan karyawan untuk meninggalkan perusahaan.
2. Menemukan model terbaik berdasarkan akurasi tertinggi untuk memprediksi kemungkinan karyawan akan meninggalkan perusahaan.

### Solution statements
Berdasarkan penjelasan di atas, terdapat beberapa solusi yang dapat diterapkan untuk mencapai tujuan proyek ini, yaitu:
1. Melakukan *Exploratory Data Analysis (EDA)* untuk mengetahui total distribusi karyawan yang meninggalkan perusahaan berdasarkan jenis kelamin, usia, pendidikan, mengidentifikasi faktor-faktor yang memengaruhi keputusan karyawan untuk keluar, dan menemukan faktor tertentu yang paling berkaitan dengan tingkat turnover tertentu.

2. Menggunakan 3 model machine learning untuk memprediksi turnover karyawan, yaitu *K-Nearest Neighbor (KNN) Algorithm*, *Decision Tree Classifier Algorithm*, dan *Random Forest Classifier Algorithm*.

3. Menggunakan *Accuracy, Precision, Recall, F1 Score* pada masing-masing model *machine learning* untuk menentukan model terbaik berdasarkan akurasi tertinggi dalam memprediksi turnover karyawan.
   
4. Menggunakan *feature importance* untuk mengevaluasi kontribusi tiap fitur dalam memprediksi potensi karyawan meninggalkan perusahaan, kemudian memvisualisasikan hasilnya dengan grafik batang.

## Data Understanding

Dataset yang digunakan untuk memprediksi tingkat turnover karyawan diambil dari platform Kaggle Dataset  [Employee](https://www.kaggle.com/datasets/tejashvi14/employee-future-prediction/data) dengan kategori *dataset* dan skor kegunaan 10/10. Dataset tersebut berisi sebuah file bernama `Employee.csv` dengan format .csv (*comma-separated values)* dan ukuran 195.25 kB.

Selanjutnya, dilakukan proses Exploratory Data Analysis (EDA) sebagai langkah awal untuk menganalisis karakteristik data, mengidentifikasi pola, anomali, serta memeriksa asumsi-asumsi dalam data menggunakan teknik statistik dan visualisasi grafis.

#### 1. Variabel pada Data
Berikut adalah penjelasan mengenai variabel-variabel yang ada pada dataset : 

![image](https://github.com/user-attachments/assets/4553f120-cfc1-40b7-b755-bd38768bc5e0)

Dari gambar diatas menunjukkan bahwa DataFrame employee memiliki 4.653 baris dan 9 kolom. Semua kolom tidak memiliki nilai yang hilang (non-null), dengan tipe data yang terdiri dari 5 kolom bertipe int64 (seperti *JoiningYear*, *PaymentTier*, *Age*, *ExperienceInCurrentDomain*, dan *LeaveOrNot*) dan 4 kolom bertipe object (seperti *Education*, *City*, *Gender*, dan *EverBenched*).

Berikut adalah keterangan untuk masing-masing variabel:
- `Education`: Level pendidikan formal tertinggi yang diperoleh oleh karyawan
- `JoiningYear`: Tahun bergabung dengan perusahaan
- `City`: Lokasi pekerjaan
- `PaymentTier`: Level pembayaran (1-3). 3 berarti gaji lebih tinggi.
- `Age`: Usia karyawan
- `Gender`: Jenis kelamin karyawan
- `EverBenched`: Apakah karyawan pernah dibekukan atau tidak
- `ExperienceInCurrentDomain`: Pengalaman kerja di perusahaan saat ini
- `LeaveOrNot`: Apakah karyawan akan meninggalkan perusahaan dalam 2 tahun ke depan

#### 2. Deskripsi Statistik
![image](https://github.com/user-attachments/assets/feacb97f-cf90-4d41-9211-8e0567750016)

Gambar tersebut memberikan ringkasan statistik untuk kolom numerik dalam DataFrame *employee*

#### 3. Menangani *Missing Value*
![image](https://github.com/user-attachments/assets/71727a66-319a-49f1-a58c-5200399f247a)

Berdasarkan gambar tersebut menunjukkan bahwa tidak ada nilai yang hilang (null) di setiap kolom DataFrame employee, karena semua kolom memiliki total nilai yang tidak kosong (0 nilai null).

#### 4. Menangani *Outliers*
Outliers adalah sampel data yang memiliki nilai yang sangat berbeda atau jauh dari sebagian besar data lainnya, yang dapat mempengaruhi atau merusak hasil analisis data. Berikut adalah boxplot yang menampilkan distribusi dan outlier dari kolom `JoiningYear`, `Age`, `PaymentTier`, dan `ExperienceInCurrentDomain` dalam DataFrame `employee`.
![outliers](https://github.com/user-attachments/assets/09855369-beec-478d-9446-bf12fa69195a)
Berikut merupakan hasil statistik deskriptif:
<table><thead><tr><th>Column</th><th>Min</th><th>Q1</th><th>Median</th><th>Q3</th><th>Max</th><th>Outliers</th></tr></thead><tbody><tr><td>JoiningYear</td><td>2012</td><td>2013.0</td><td>2015.0</td><td>2017.0</td><td>2018</td><td>0</td></tr><tr><td>PaymentTier</td><td>1</td><td>3.0</td><td>3.0</td><td>3.0</td><td>3</td><td>1161</td></tr><tr><td>Age</td><td>22</td><td>26.0</td><td>28.0</td><td>32.0</td><td>41</td><td>0</td></tr><tr><td>ExperienceInCurrentDomain</td><td>0</td><td>2.0</td><td>3.0</td><td>4.0</td><td>7</td><td>0</td></tr></tbody></table>

Berdasarkan hasil statistik deskriptif yang diperoleh, beberapa kesimpulan dapat diambil mengenai dataset yang dianalisis. Berikut adalah analisis untuk setiap kolom:

1. JoiningYear:
	 - Rentang Tahun : Data mencakup tahun dari 2012 hingga 2018, dengan Q1 pada 2013 dan median di 2015.
	 - Outliers: Tidak ditemukan outlier, yang menunjukkan bahwa distribusi data cukup konsisten.
2. PaymentTier:
	- Rentang: Nilai payment tier bervariasi antara 1 hingga 3, dengan mayoritas data berada pada tier 3, terlihat dari Q1, median, dan Q3 yang semuanya bernilai 3.
	- *Outliers*: Teridentifikasi 1161 outlier yang cukup signifikan. Hal ini menunjukkan adanya sejumlah data yang berada di luar rentang normal, yang mungkin memerlukan investigasi lebih lanjut.
3. Age:
	- *Rentang Usia*: Usia karyawan berkisar antara 22 hingga 41 tahun, dengan median pada usia 28 tahun dan Q1 di 26 tahun, yang menunjukkan bahwa sebagian besar karyawan berada pada usia dewasa muda hingga paruh baya.
	- *Outliers*: Tidak ditemukan outlier, yang menandakan distribusi usia yang relatif normal.
4. ExperienceInCurrentDomain:
	- *Rentang Pengalaman*: Pengalaman karyawan bervariasi antara 0 hingga 7 tahun, dengan median di 3 tahun, yang menunjukkan banyaknya karyawan yang relatif baru dalam domain pekerjaan mereka.
	- *Outliers*: Tidak ada outlier terdeteksi, menandakan distribusi pengalaman yang stabil.
	
*Outlier* pada `PaymentTier`: Mengingat adanya 1161 outlier dalam kolom `PaymentTier`, dapat dilihat bahwa `PaymentTier` memiliki lebih banyak outlier. Di sini, diputuskan untuk mempertahankan outlier dalam dataset klasifikasi `LeaveOrNot` karena data tersebut mencerminkan kondisi nyata dalam struktur organisasi. Dalam analisis yang dilakukan, terlihat bahwa jumlah karyawan pada level pembayaran yang lebih rendah jauh lebih banyak dibandingkan dengan level yang lebih tinggi. Ini adalah representasi yang valid dari situasi yang ada di perusahaan, di mana mayoritas karyawan berada di tier pembayaran yang lebih rendah. Hal ini juga dipengaruhi oleh kenyataan bahwa jabatan tinggi biasanya memiliki tingkat pembayaran yang lebih tinggi, namun jabatan-jabatan tersebut sangat sedikit jumlahnya di setiap perusahaan, sementara sebagian besar posisi diisi oleh karyawan di tingkat staff yang memiliki tier pembayaran lebih rendah.

Mempertahankan outliers memungkinkan model untuk mempertimbangkan semua aspek yang berkontribusi terhadap keputusan karyawan. Jika ada faktor-faktor tertentu yang menyebabkan karyawan pada level bawah memilih untuk meninggalkan perusahaan, model harus memiliki data tersebut untuk menganalisis dan memberikan wawasan yang berharga bagi manajemen

#### 5. *Univariate Analysis*
Melakukan analisis data univariate pada fitur-fitur numerik untuk memahami distribusi dan pola data. Proses ini dilakukan dengan memanfaatkan visualisasi histogram untuk setiap fitur numerik, yang dapat memberikan gambaran yang lebih jelas mengenai sebaran data.
![Univariate Analysis](https://github.com/user-attachments/assets/6d7161ac-bc95-4f40-bbf4-d6b9717c7723)

Interpretasi : 
1. Distribusi Usia Karyawan
Berdasarkan distribusi usia karyawan yang terbagi dalam beberapa rentang usia, terlihat kecenderungan bahwa sebagian besar karyawan berada pada rentang usia muda, dengan jumlah yang lebih banyak di beberapa rentang usia tertentu. Beberapa rentang usia menunjukkan konsentrasi yang lebih tinggi, sementara rentang usia yang lebih tua cenderung memiliki jumlah yang lebih sedikit. Hal ini mengindikasikan bahwa perusahaan kemungkinan besar memiliki proporsi karyawan yang lebih banyak di awal karier mereka, sementara jumlah karyawan yang lebih tua relatif lebih rendah.
2. Distribusi *LeaveOrNot*
Distribusi data pada kolom *LeaveOrNot* menunjukkan bahwa kurang lebih 3000 karyawan tidak meninggalkan perusahaan, sementara 1600 karyawan memutuskan untuk meninggalkan perusahaan. Dari data ini, dapat disimpulkan bahwa sebagian besar karyawan memilih untuk tetap bertahan di perusahaan. Meskipun demikian, angka karyawan yang memilih untuk meninggalkan perusahaan cukup signifikan, yang menunjukkan bahwa meskipun banyak yang bertahan, perusahaan juga mengalami tingkat perputaran karyawan yang tidak bisa diabaikan.
3. Jumlah Karyawan Berdasarkan Kota
Distribusi jumlah karyawan berdasarkan kota menunjukkan bahwa Bangalore memiliki jumlah karyawan terbanyak, diikuti oleh Pune, dan New Delhi . Hal ini mengindikasikan bahwa Bangalore merupakan lokasi dengan konsentrasi karyawan yang paling tinggi, yang kemungkinan besar merupakan pusat operasional atau kantor utama perusahaan. Sementara itu, Pune dan New Delhi memiliki jumlah karyawan yang lebih sedikit, meskipun masih tergolong signifikan.

#### 6. *Multivariate Analysis*
Melakukan visualisasi distribusi data untuk fitur-fitur numerik dalam dataframe *employee*. Visualisasi ini menggunakan library seaborn dengan fungsi *pairplot*, dengan parameter `diag_kind='kde'` untuk menggambarkan estimasi distribusi probabilitas antar fitur numerik berdasarkan kolom *LeaveOrNot*.
![Multivariate Analysis](https://github.com/user-attachments/assets/c8196259-fd57-4e6e-a42a-05df69d02487)

Distribusi *LeaveOrNot* Berdasarkan Usia, *Gender*, dan Pendidikan

![Multivariate Analysis 2](https://github.com/user-attachments/assets/d818dd17-796a-45f3-84bd-15b4f58b11bd)
Interpretasi :
1. Statistik Usia Berdasarkan Status Meninggalkan Perusahaan:
Rata-rata usia karyawan yang meninggalkan perusahaan sedikit lebih rendah dibandingkan dengan mereka yang tetap tinggal, meskipun perbedaannya tidak signifikan.
2. Distribusi Berdasarkan Gender:
Lebih banyak karyawan laki-laki yang tetap tinggal di perusahaan, sementara karyawan perempuan lebih banyak yang meninggalkan perusahaan.
3. Distribusi Berdasarkan Pendidikan:
Karyawan dengan tingkat pendidikan lebih tinggi (Magister dan Doktor) cenderung lebih banyak meninggalkan perusahaan dibandingkan dengan mereka yang berpendidikan Sarjana.

Perbandingan Keputusan Meninggalkan Perusahaan Berdasarkan *Gender* dan *Payment Tier* 

![Multivariate Analysis 3](https://github.com/user-attachments/assets/b8c06a88-d5d6-4e55-a337-ddfa9ee31a1d)

Interpretasi : 
1. Pada *Payment Tier 1* (pekerjaan dengan gaji lebih rendah), lebih banyak karyawan perempuan yang meninggalkan perusahaan dibandingkan dengan karyawan laki-laki. Sementara itu, lebih banyak karyawan laki-laki yang memilih untuk tetap bekerja di perusahaan.
2. Pada *Payment Tier 2*, sekali lagi, lebih banyak karyawan perempuan yang meninggalkan perusahaan dibandingkan dengan karyawan laki-laki. Hal ini menunjukkan bahwa karyawan laki-laki cenderung lebih bertahan dan tetap bekerja di perusahaan.
3. Namun, pada *Payment Tier 3*, laki-laki menunjukkan pola yang berbeda, dengan lebih banyak yang berpindah pekerjaan dibandingkan dengan rekan perempuan mereka.

#### 7. *Correlation Matrix *menggunakan* Heatmap*
Melakukan analisis korelasi antar fitur numerik dengan memanfaatkan visualisasi heatmap dari matriks korelasi. Visualisasi ini memungkinkan untuk mengidentifikasi hubungan antara berbagai variabel numerik secara visual, sehingga memudahkan dalam mendeteksi pola-pola penting yang dapat mempengaruhi hasil analisis atau model
![Correlation Matrix](https://github.com/user-attachments/assets/96aeaf22-9d11-4e81-99f7-f3b05891d622)

Berikut adalah hasil analisis korelasi matriks yang dapat disimpulkan:
1. LeaveOrNot (Meninggalkan Perusahaan):
	- *JoiningYear* dan *LeaveOrNot*: Terdapat korelasi positif sebesar 0.18, yang menunjukkan bahwa karyawan yang baru bergabung cenderung memiliki kemungkinan lebih besar untuk meninggalkan perusahaan. Hal ini mengindikasikan bahwa karyawan yang baru bergabung mungkin merasa kurang terikat atau memiliki lebih banyak peluang di luar perusahaan.
	- *PaymentTier* dan *LeaveOrNot*: Korelasi negatif sebesar -0.19 menunjukkan bahwa karyawan dengan tingkat pembayaran lebih tinggi cenderung lebih jarang meninggalkan perusahaan. Ini bisa menandakan bahwa kompensasi yang lebih baik meningkatkan kepuasan kerja dan motivasi untuk tetap bertahan.
2. *Variabel Lain*:
	-	*Age* dan *ExperienceInCurrentDomain*: Korelasi negatif antara `Age` dan `ExperienceInCurrentDomain` sebesar -0.13 menunjukkan bahwa karyawan yang lebih tua cenderung memiliki pengalaman lebih banyak di berbagai domain atau sering berpindah pekerjaan.
3. Interpretasi Umum:
	- Secara keseluruhan, variabel *JoiningYear* dan *PaymentTier* tampaknya menjadi faktor penting dalam keputusan karyawan untuk meninggalkan perusahaan. Karyawan yang baru bergabung atau yang memiliki tingkat pembayaran lebih rendah cenderung lebih besar kemungkinan untuk meninggalkan perusahaan.

## Data Preparation

Pada tahap persiapan data (data preparation), langkah-langkah ini sangat penting untuk memastikan data yang digunakan siap dan dapat memaksimalkan kinerja model machine learning. Proses ini melibatkan pembersihan dan transformasi data agar dapat memberikan hasil yang optimal saat digunakan untuk pelatihan model. Berikut adalah dua tahapan utama dalam persiapan data yang dilakukan, yaitu:

#### 1. *Label Encoding*
Label Encoding digunakan karena setiap kategori dalam kolom akan diubah menjadi nilai numerik yang sesuai.
Ini bertujuan untuk mempersiapkan data dengan mengonversi kolom-kolom kategorikal dalam dataset menjadi format numerik, yang diperlukan untuk digunakan dalam model machine learning. Adapun langkah-langkah yang dilakukan adalah sebagai berikut:
1.  Konversi Kolom Kategorikal Menjadi Kode Numerik: Kolom-kolom dengan data kategorikal seperti  `Education`,  `City`,  `Gender`, dan  `EverBenched`  diubah menjadi tipe data kategorikal, kemudian setiap kategori tersebut dikodekan ke dalam bentuk numerik. Hal ini dilakukan karena algoritma machine learning umumnya memerlukan data dalam format numerik agar dapat diolah dengan baik.
   ![Label Encoding](https://github.com/user-attachments/assets/b28e8ae6-d973-4188-a90c-1849cc668d16)
   
    Terlihat bahwa pada gambar beberapa kolom kategorikan akan di konversi menjadi bentuk numerik, perbedaan setelah dan sesudahnya.

1.  Verifikasi Hasil Konversi: Setelah konversi, tipe data untuk setiap kolom diperiksa untuk memastikan bahwa perubahan ke format numerik telah berhasil dilakukan. 
![encoding](https://github.com/user-attachments/assets/1fbc8fe9-c3bf-484a-bcc3-0a14694136a7)

#### 2. *Split Data*

Menentukan variabel `x` yang berisi fitur-fitur yang digunakan untuk memprediksi keputusan meninggalkan perusahaan (*LeaveOrNot*), dengan mengecualikan kolom `LeaveOrNot` sebagai fitur, serta variabel `y` yang merupakan target atau nilai yang akan diprediksi, yaitu kolom `LeaveOrNot`.

    #split data
    X = df.drop('LeaveOrNot', axis = 1)
    y = df['LeaveOrNot']


Melakukan pembagian dataset menggunakan `train_test_split` untuk memisahkan data menjadi data latih (*training*) dan data uji (*testing*), dengan proporsi 70% untuk data latih dan 30% untuk data uji. Kemudian, menampilkan jumlah total data, serta jumlah data latih dan data uji.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
Selanjutnya, diperoleh hasil pembagian data untuk setiap kategori, yaitu sebagai berikut:
    
    Dimensi data training (X_train): (3257, 8) 
    Dimensi target training (y_train): (3257,) 
    Dimensi data testing (X_test): (1396, 8) 
    Dimensi target testing (y_test): (1396,)

## Modeling
Terdapat 3 algoritma Machine Learning yang diterapkan untuk membangun model, yaitu sebagai berikut.
#### 1. *K-Nearest Neighbor (KNN) Algorithm*
*K-Nearest Neighbor (KNN) Algorithm* digunakan untuk klasifikasi dan regresi berdasarkan jarak terdekat antara data baru dan data yang sudah ada dalam dataset pelatihan. Untuk klasifikasi, KNN mengidentifikasi kk tetangga terdekat dari data baru dan memberikan label yang paling umum di antara tetangga-tetangga tersebut sebagai prediksi. Untuk regresi, KNN menghitung rata-rata nilai dari kk tetangga terdekat untuk memprediksi nilai data baru [[3]](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00973-y).

Dalam penerapannya, KNN digunakan melalui `KNeighborsClassifier` dari library `sklearn.neighbors`, di mana data `X_train` dan `y_train` dipakai untuk melatih model, sedangkan `X_test` dan `y_test` digunakan untuk evaluasi model dengan data uji. Parameter `n_neighbors` menentukan jumlah tetangga terdekat yang akan dipertimbangkan, dan dalam proyek ini, nilai `n_neighbors = 5` dipilih untuk memprediksi apakah seorang karyawan akan meninggalkan perusahaan atau tidak.

#### 2. *Decision Tree Classifier Algorithm*
*Decision Tree Classifier Algorithm* ini membangun model berupa pohon keputusan (decision tree) yang menentukan nilai target berdasarkan nilai atribut dari data. Setiap node dalam pohon keputusan adalah pengelompokan berdasarkan nilai atribut tertentu, dan cabang-cabangnya mengarah ke node anak berdasarkan kriteria tertentu. Decision tree mudah dipahami dan dapat menggambarkan pola yang kompleks dalam data [[4]](https://link.springer.com/article/10.1007/s10462-011-9272-4).

Dalam proyek ini, *Decision Tree Classifier* digunakan untuk memprediksi apakah seorang karyawan akan meninggalkan perusahaan atau tidak. Model ini diimplementasikan menggunakan `DecisionTreeClassifier` dari library `sklearn.tree`, dengan parameter `max_depth` yang diatur ke 3 untuk membatasi kedalaman pohon hingga tiga tingkat. Ini bertujuan untuk menjaga model agar tidak terlalu kompleks dan menghindari *overfitting*.

Parameter `max_depth=3` membatasi kedalaman maksimal pohon, yang membantu menjaga keseimbangan antara akurasi dan kompleksitas model.

#### 3. *Random Forest Classifier Algorithm*

*Random Forest Classifier Algorithm*  merupakan metode ensemble learning yang menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting. Algoritma ini bekerja dengan cara membuat banyak pohon keputusan dari subset data yang berbeda, dan hasil akhirnya diambil berdasarkan mayoritas voting (untuk klasifikasi) atau rata-rata (untuk regresi). Random Forest sangat efektif dalam mengatasi masalah dataset yang besar dan kompleks serta data yang memiliki banyak fitur[[5]](https://link.springer.com/article/10.1023/A:1010933404324).

Dalam proyek ini, *Random Forest Classifier* digunakan untuk memprediksi apakah seorang karyawan akan meninggalkan perusahaan atau tidak. Model ini diimplementasikan menggunakan `RandomForestClassifier` dari library `sklearn.ensemble`, dengan beberapa parameter yang disesuaikan, seperti `random_state` untuk memastikan reprodusibilitas, `max_features='sqrt'` untuk membatasi jumlah fitur yang akan digunakan di setiap pohon secara acak, `n_jobs=1` untuk menentukan jumlah CPU yang digunakan, dan `verbose=1` untuk menampilkan log proses saat pelatihan.

Parameter `max_features='sqrt'` memungkinkan setiap pohon dalam hutan untuk menggunakan akar kuadrat dari total fitur yang tersedia, yang membantu menciptakan pohon yang lebih beragam.


Ketiga model yang telah dibuat di atas, yaitu model dengan algoritma *K-Nearest Neighbors (KNN), Decision Tree, dan Random Forest*, akan diuji kinerjanya masing-masing. Pengujian ini bertujuan untuk menentukan model mana yang memberikan hasil prediksi paling akurat dengan tingkat kesalahan yang paling rendah di antara ketiganya.

## Evaluation

Dalam proyek ini, penilaian kinerja model dilakukan menggunakan metrik evaluasi *Accuracy*, *Precision*, *Recall*, dan *F1 Score* untuk setiap model yang dibangun. Sebelum itu, akan dijelaskan terlebih dahulu cara menghitung masing-masing metrik tersebut.

*Confusion matrix* adalah untuk memvisualisasikan dan menganalisis hasil prediksi yang dibuat oleh model

![matrix](https://github.com/user-attachments/assets/5a4156fb-362a-4a18-bbc0-7bc0aede88f0)

Pada Confusion matrix menunjukkan nilai aktual, sedangkan setiap kolomnya menunjukkan nilai prediksi. Matriks ini berisi empat kategori, yaitu TP, TN, FP, dan FN, seperti terlihat pada gambar [[6]](https://ilmudatapy.com/apa-itu-confusion-matrix/).

- *True Positive (TP)*: Jumlah data yang benar-benar positif dan diprediksi sebagai positif.
- *True Negative (TN)*: Jumlah data yang benar-benar negatif dan diprediksi sebagai negatif.
- *False Positive (FP)*: Data sebenarnya negatif, tetapi diprediksi sebagai positif
- *False Negative (FN)*: Data sebenarnya positif, tetapi diprediksi sebagai negatif.

Dalam analisis kinerja model klasifikasi, metrik evaluasi yang digunakan untuk mengukur performa model adalah *Accuracy*, *Precision*, *Recall*, dan *F1 Score*. Metrik-metrik ini dihitung berdasarkan nilai-nilai yang diperoleh dari  Confusion matrix, yang terdiri dari empat kategori utama, yaitu True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN). Berikut adalah penjelasan masing-masing metrik:

- *Accuracy*: Mengukur proporsi prediksi yang benar (baik positif maupun negatif) dibandingkan dengan seluruh prediksi yang dilakukan oleh model.
  
  ![accuracy](https://github.com/user-attachments/assets/474a06b7-a625-43a9-90a0-db3caefe55fa)
- *Precision*: Mengukur ketepatan model dalam memprediksi kelas positif, yaitu rasio antara True Positives dengan total data yang diprediksi sebagai positif.
  
  ![precision](https://github.com/user-attachments/assets/c5f8cf98-eab1-4f9a-8911-1c3e688cb4a6)
- *Recall*: Mengukur sejauh mana model dapat mendeteksi seluruh data aktual yang positif, yaitu rasio antara True Positives dengan total data aktual yang positif.
  
  ![Recall](https://github.com/user-attachments/assets/83247d27-03b8-4608-8179-425cb7618a7c)
- *F1 Score*: Merupakan rata-rata harmonis dari Precision dan Recall, yang memberikan gambaran menyeluruh mengenai keseimbangan antara ketepatan dan kelengkapan prediksi model.
  
  ![F1](https://github.com/user-attachments/assets/1ded5f88-b272-422f-a3e8-968d5e264fa6)


Penerapan metrik _Accuracy_, _Precision_, _Recall_, dan _F1 Score_ dilakukan pada ketiga model klasifikasi, yaitu K-Nearest Neighbors (KNN), Decision Tree, dan Random Forest, untuk mengevaluasi kinerja masing-masing model
![evaluation model](https://github.com/user-attachments/assets/2bbbd973-6bce-44a6-9fe8-2078a259bb3c)

Dari gambar diatas menunjukkan hasil evaluasi tiga model klasifikasi: K-Nearest Neighbors (KNN), Decision Tree, dan Random Forest, dengan metrik *Accuracy*, *Precision*, *Recall*, dan *F1 Score*.
- *Random Forest* menunjukkan performa terbaik dengan akurasi (82.66%), precision (82.38%), recall (82.66%), dan F1 score (82.33%).
- *Decision Tree* berada di posisi kedua, dengan akurasi 78.01%, precision 78.27%, recall 78.01%, dan F1 score 76.27%.
- *K-Nearest Neighbors* memiliki akurasi terendah (76.36%) dibandingkan model lainnya.

Berdasarkan akurasi yang diperoleh dari ketiga model, dibuat sebuah diagram box plot untuk membandingkan nilai akurasi masing-masing model seperti yang terlihat pada gambar berikut.
![Comparison of Algorithm](https://github.com/user-attachments/assets/aeb979ae-ade3-48da-be38-9992358d4811)

Kesimpulannya, model *Random Forest* memberikan performa terbaik dengan akurasi 82.66%, diikuti oleh *Decision Tree* dengan akurasi 78.01%, dan *K-Nearest Neighbors* yang memiliki akurasi terendah 76.36%. Oleh karena itu, model yang dipilih untuk memprediksi apakah karyawan akan meninggalkan perusahaan atau tidak adalah *Random Forest*.

Untuk mengevaluasi kontribusi tiap fitur dalam memprediksi potensi karyawan meninggalkan perusahaan, dapat menggunakan nilai *feature importance* yang dihitung oleh model terbaik yang telah dilatih. 

    # Mendapatkan feature importance
    feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

Kode tersebut akan menghasilkan urutan fitur berdasarkan pengaruhnya menghubungkannya dengan data training `X_train`, dan mengurutkannya dari yang paling tinggi hingga paling kecil.

![feature importance](https://github.com/user-attachments/assets/cdd6b244-615f-4118-aec0-f5625f7e3804)

Model ini memberikan skor untuk setiap fitur, yang mencerminkan seberapa besar kontribusi fitur tersebut terhadap keputusan prediksi.

Dari seluruh skor setiap fitur yang diketahui, dibentuk bar plot untuk melihat perbandingan nilai akurasi model sebagai berikut.

![feature importance 2](https://github.com/user-attachments/assets/3c90cee4-bab5-45c0-b583-491d2fe20f5f)


Berdasarkan gambar di atas serta dari data `feature_importances` untuk fitur *JoiningYear* (skor 0.328431) memiliki pengaruh terbesar terhadap kemungkinan seorang karyawan untuk meninggalkan perusahaan. Ini menunjukkan bahwa karyawan yang baru bergabung dengan perusahaan memiliki potensi lebih tinggi untuk pergi, yang bisa disebabkan oleh ketidakpuasan atau ketidaknyamanan di awal masa kerja. *Age* dan *City* juga cukup berpengaruh dengan skor 0.163789 dan 0.113360, yang mungkin mengindikasikan bahwa faktor demografi mempengaruhi tingkat retensi karyawan

Namun, fitur *EverBenched* memiliki pengaruh yang sangat kecil (skor 0.018730), yang mungkin menunjukkan bahwa pengalaman karyawan terkait dengan ketidakaktifan atau penempatan tidak terlalu memengaruhi keputusan mereka untuk bertahan.

Dari analisis tersebut, melihat bahwa *JoiningYear* memiliki pengaruh terbesar terhadap kemungkinan karyawan untuk meninggalkan perusahaan. Hal ini relevan dengan tujuan perusahaan untuk mengurangi *turnover* di kalangan karyawan baru. Dengan mengetahui bahwa karyawan yang baru bergabung memiliki potensi lebih tinggi untuk meninggalkan perusahaan, perusahaan dapat lebih fokus dalam merancang program orientasi dan integrasi yang lebih baik bagi karyawan baru.

## Kesimpulan
1. *JoiningYear* dan *Age* adalah dua fitur utama yang menunjukkan pengaruh paling besar terhadap potensi karyawan untuk meninggalkan perusahaan. Ini berarti bahwa pengalaman karyawan di perusahaan (seperti lama bekerja) serta faktor usia menjadi indikator penting dalam strategi retention.

2. Untuk memaksimalkan hasil retention, perusahaan bisa melakukan intervensi yang lebih spesifik pada karyawan dengan masa kerja yang lebih singkat atau yang berada pada rentang usia tertentu.
   
3. Setelah menguji data menggunakan 3 model machine learning, yaitu  *K-Nearest Neighbors (KNN)*, *Decision Tree*, dan *Random Forest*. Secara keseluruhan, *Random Forest* menunjukkan kinerja terbaik di antara ketiga model.

## Referensi

[1] Gallup, "Workplace Trends Leaders Should Watch in 2024," Gallup Workplace Blog. [Online]. Available: [https://www.gallup.com/workplace/547283/workplace-trends-leaders-watch-2024.aspx.](https://www.gallup.com/workplace/547283/workplace-trends-leaders-watch-2024.aspx) [Accessed: 06-Nov-2024].

[2] Award, "Average Employee Turnover Rates by Industry," Award Blog. [Online]. Available: [https://www.award.co/blog/employee-turnover-rates.](https://www.award.co/blog/employee-turnover-rates) [Accessed: 06-Nov-2024].

[3] R. K. Halder, M. N. Uddin, M. A. Uddin, S. Aryal, and A. Khraisat, "Enhancing K-nearest neighbor algorithm: a comprehensive review and performance analysis of modifications," Journal of Big Data, vol. 2024, no. 1, pp. 1-15, 2024.

[4] S. B. Kotsiantis, "Decision trees: a recent overview," Artificial Intelligence Review, vol. 39, no. 3, pp. 261-283, 2013.

[5] L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.

[6] A, Lutfia, "Apa Itu Confusion Matrix?," [ilmudatapy.com.](https://ilmudatapy.com) [Online]. Available: [https://ilmudatapy.com/apa-itu-confusion-matrix/.](https://ilmudatapy.com/apa-itu-confusion-matrix/) [Accessed: 06-Nov-2024]