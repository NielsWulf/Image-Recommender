
Skip to content
Navigation Menu

    NielsWulf
    /
    Image-Recommender

Code
Issues
Pull requests
Actions
Projects
Wiki
Security

    Insights

Editing README.md in Image-Recommender
Breadcrumbs

    Image-Recommender

/
in
main

Indent mode
Indent size
Line wrap mode
Editing README.md file contents
Selection deleted
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
    kernel_pca_sample_size: Number of samples to use for Kernel PCA (default is 10,000).

This Python script performs dimensionality reduction on a set of image embeddings using both Incremental PCA and Kernel PCA. The reduced embeddings are then saved to pickle files for further analysis.

- **Incremental PCA**: Reduces the dimensionality of large datasets using Incremental PCA, which processes the data in batches, making it suitable for handling large-scale data.
- **Kernel PCA**: Further reduces the dimensionality of the embeddings using Kernel PCA on a subsampled dataset to capture non-linear relationships.
- **Batch Processing**: Efficiently handles large datasets by processing data in batches during Incremental PCA.
- **Subsampling for Kernel PCA**: Allows for the selection of a random subset of the data to make Kernel PCA computationally feasible.
- **Pickle Data Storage**: Stores the reduced embeddings and trained models in pickle files for easy retrieval and further use.


### 5) Similarity Measures

#### similarity_Measure_Color_Profile.py : 

```bash
similarity_Measure_Color_Profile.py
```

This Python script compares the color histograms of a target image (or a set of images) against a large collection of precomputed histograms stored in a pickle file. It identifies and displays the most visually similar images based on the selected comparison method.


Comparison Method: You can change the method parameter in compare_histograms to use different similarity metrics (e.g., correlation, chi-square, intersection, and Bhattacharyya).
We reccomend using correlation or Bhattacharyya



## Results from correlation


![grafik](https://github.com/user-attachments/assets/0cc2ae16-7ac9-4118-9721-e9d25cb55605)


## Results from Bhattacharyya


![grafik](https://github.com/user-attachments/assets/82a0aa8d-473e-4076-abdc-29c17ca8c8d4)


#### similarity_Measure_Embeddings.py: 

```bash
similarity_Measure_Embeddings.py
```

This Python script uses a pre-trained ResNet model to generate image embeddings for the input and compares them to a database of precomputed embeddings to find the most visually similar images. 

Comparison Method: The similarity is measured using cosine similarity, which is ideal for comparing high-dimensional vectors like image embeddings.

Top N Results: Modify the top_n parameter to control how many of the most similar images are returned and displayed.

## Resluts from Embeddings

![grafik](https://github.com/user-attachments/assets/0ab82704-fb16-4b74-866b-ff1e8d5c8f0c)





#### similarity_Measure_PCA_Embeddings.py:

```bash
similarity_Measure_PCA_Embeddings.py
```

Execute the script to compute the embedding of an input image, apply PCA, and find the most similar images in the dataset that was also dimension reduced.




## Results for PCA Embeddings 75 dimensions: 


![grafik](https://github.com/user-attachments/assets/713ba37a-a117-484f-a0a8-a8086236dc9d)



## Results for PCA Embeddings 100 dimensions: 

![grafik](https://github.com/user-attachments/assets/44d216ea-9efb-41ee-a68f-1d1a3ecca1b9)


