import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def read_csv(file_path):
    return pd.read_csv(file_path)

def update_graph(frame):
  
    try:
      df = read_csv(csv_file_path)
      
      epochs = []
      train_accuracy_means = []
      train_loss_means = []
      val_accuracy_uniques = []
      val_loss_uniques = []

      for epoch in df['epoch'].unique():
          dados_epoca = df[df['epoch'] == epoch]
          
          train_accuracy_mean = dados_epoca['train_accuracy'].mean()
          train_loss_mean = dados_epoca['train_loss'].mean()
          val_accuracy_unique = dados_epoca['val_accuracy'].mean()
          val_loss_unique = dados_epoca['val_loss'].mean()
          
          epochs.append(epoch)
          train_accuracy_means.append(train_accuracy_mean)
          val_accuracy_uniques.append(val_accuracy_unique)
          train_loss_means.append(train_loss_mean)
          val_loss_uniques.append(val_loss_unique)
      
      resultados = pd.DataFrame({
          'epoch': epochs,
          'train_accuracy': train_accuracy_means,
          'val_accuracy': val_accuracy_uniques,
          'train_loss': train_loss_means,
          'val_loss': val_loss_uniques
      })
      
      ax1.clear()
      ax2.clear()

      ax1.plot(resultados['epoch'], resultados['train_accuracy'], label='Train Accuracy')
      ax1.plot(resultados['epoch'], resultados['val_accuracy'], label='Validation Accuracy')
      ax1.set_xlabel('Epoch')
      ax1.set_ylabel('Accuracy')
      ax1.set_title('Accuracy vs. Epoch')
      ax1.legend()

      ax2.plot(resultados['epoch'], resultados['train_loss'], label='Train Loss')
      ax2.plot(resultados['epoch'], resultados['val_loss'], label='Validation Loss')
      ax2.set_xlabel('Epoch')
      ax2.set_ylabel('Loss')
      ax2.set_title('Loss vs. Epoch')
      ax2.legend()

      plt.subplots_adjust(bottom=0.25)
      plt.draw()
      plt.pause(1)
    except:
      pass
      

csv_file_path = './lightning_logs/csv_file/version_0/metrics.csv'

plt.figure(figsize=(15, 3))
# plt.figure(figsize=(10, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

ani = animation.FuncAnimation(plt.gcf(), update_graph, interval=5000)  # Atualiza a cada 5 segundos

plt.show()
