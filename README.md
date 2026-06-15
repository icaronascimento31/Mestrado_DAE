# Mestrado_DAE

Projeto de pesquisa de mestrado voltado para a mitigação dos efeitos de **crosstalk eletrônico** em calorímetros utilizando técnicas de **Deep Learning**, com foco em **Convolutional Denoising Autoencoders (DAEs)**.

O objetivo principal é reconstruir sinais de energia e tempo a partir de leituras contaminadas por crosstalk e ruído, preservando as características físicas originais dos eventos detectados.

---

## Objetivos

- Mitigar os efeitos de crosstalk em sinais de calorímetros.
- Reconstruir distribuições de energia e tempo utilizando redes neurais convolucionais.
- Avaliar o desempenho da reconstrução por meio de métricas estatísticas e físicas.
- Comparar os sinais reconstruídos com os sinais verdadeiros (ground truth).

---

## Estrutura do Projeto

```text
Mestrado_DAE/
│
├── Notebooks/
│   ├── DAE_CONV_Samples.ipynb
│   ├── DAE_Conv_OptFilt.ipynb
│   ├── extract_data.ipynb
│   ├── extract_histo_rmse.ipynb
│   ├── extract_loss.ipynb
│   └── extract_r2.ipynb
│
├── Scripts/
│   ├── DAE_XT_Energy.py
│   ├── DAE_XT_Time.py
│   └── utils.py
│
├── results/
│   ├── boxplot_energy.png
│   ├── curvas_medias.png
│   ├── k_fold_energy.png
│   ├── output_energy.png
│   └── ...
│
├── requirements.txt
├── LICENSE
└── README.md
```

---
