import pss_model_creator
import preparing_model

import torch
import torch.utils.data as data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = preparing_model.MODEL

test_iterator = data.DataLoader(test_data, 
                                    batch_size = preparing_model.BATCH_SIZE)

def get_predictions(model, iterator):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()

    prev_page_images = []
    curr_page_images = []
    next_page_images = []
    
    labels = []
    probs = []
    file_names = []

    with torch.no_grad():

        for (x, y, name) in iterator:

            prev_page, curr_page, next_page = x
            prev_page = prev_page.to(device)
            curr_page = curr_page.to(device)
            next_page = next_page.to(device)
            
            y_pred = model(prev_page, curr_page, next_page)
            
            #concat_images(prev_page.cpu(), curr_page.cpu(), next_page.cpu())

            prev_page_images.append(prev_page.cpu())
            curr_page_images.append(curr_page.cpu())
            next_page_images.append(next_page.cpu())
            #labels.append(y.cpu()//2)
            labels.append(torch.div(y.cpu(), 2, rounding_mode='trunc'))
            probs.append(y_pred.cpu())
            file_names.append(name)

    prev_page_images = torch.cat(prev_page_images, dim = 0)
    curr_page_images = torch.cat(curr_page_images, dim = 0)
    next_page_images = torch.cat(next_page_images, dim = 0)    
    
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)
    file_names = list(sum(file_names, ()))

    return [prev_page_images, curr_page_images, next_page_images] , labels, probs, file_names

images, labels, probs, file_names = get_predictions(model, test_iterator)

pred_labels = torch.div(torch.argmax(probs, 1), 2, rounding_mode='trunc')

def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize = (10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels = classes);
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    plt.xticks(rotation = 20)
    plt.savefig('./model/'+ preparing_model.EXPERIMENT + '_confusion_matrix.png')

classes = {'NextPage':0, 'FirstPage':1}
plot_confusion_matrix(labels, pred_labels, classes)

report_file_path = './report_final.json'
export_report(preparing_model.EXPERIMENT, 
              labels, 
              pred_labels, 
              list(classes.keys()),
              valid_loss, 
              valid_acc_2, 
              valid_kappa_2,               
              test_loss, 
              test_acc_2, 
              test_kappa_2, 
              report_file_path)