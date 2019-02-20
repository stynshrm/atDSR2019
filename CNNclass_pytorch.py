import torch
import torch.nn as nn

# A simple network from nn.Module
class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        
        #RGB image 3 channels, 12 kernels (3x3), stride and padding 1 preserves image dimensions
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,12, kernel_size=3, stride=1, padding=1), #(in_channels=3, out_channels=12
            nn.ReLU()
            )
        
        #layer2 in_channels = layer1 out_channels 
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride =2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(24,24, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            )
        #image size = 1 maxpool reduce by half 
        # so if input is 28*28, out is 14*14, 24 is output of layer3-> 14*14*24
        self.fc = nn.Linear(in_features=14 * 14 * 24, out_features=num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)  #reshape for fc
        out = self.fc(out)
        return out

def main():
    image = torch.randn(32, 3, 28, 28)
    cnn = SimpleNet(10)
    output = cnn(image)
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    main()
    
    
    

