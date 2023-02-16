class Convolution_Layer():
    
    def __init__(self,input_shape , filter_size ,depth, bias=True, stride=1, padding=0, dilation=1):
        kernel_size=filter_size[2]
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.normal(loc=0,scale=0.0001,size=self.kernels_shape)
    
    def corelate(self,A,B):
    
        R,C=A.shape
        r,c=B.shape
        res=np.zeros((R-r+1,C-c+1))
        for i in range(R-r+1):
            for j in range(C-c+1):
                res[i][j]=np.sum(A[i:i+r,j:j+c]*B)
        return res



    def convlolve_full(self,A,B):
        R,C=A.shape
        r,c=B.shape
        #print(r,c)
        A = np.pad(A, (r-1,c-1), constant_values=(0))
        B=np.rot90(np.rot90(B))

        return corelate(A,B)

    def forward(self,inp):
        self.input = inp
        #print(self.output_shape)
        self.output = np.zeros(self.output_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += self.corelate(self.input[j], self.kernels[i, j])
        return self.output
    
    
    def backward(self, output_grad,learning_rate):
        
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = self.corelate(self.input[j], output_grad[i])
                input_gradient[j] += self.convlolve_full(output_grad[i], self.kernels[i, j])

        self.kernels -= learning_rate * kernels_gradient
        
        return input_gradient
    
    def set_weights(self, new_weights):
        self.kernels=new_weights
    



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file = 'cifar-10-batches-py/data_batch_1'
data_batch_1 = unpickle(file)


image = data_batch_1[b'data'][1].reshape(3,32,32).transpose(1,2,0)
plt.imshow(image)
image=image.transpose(2,0,1)




def make_gaussian_blur(var=2,filter_size=(3,5,5)):
    d,h,w=filter_size
    fil=np.zeros(filter_size)
    for i in range(d):
        for j in range(h):
            for k in range(w):
                p=(-1/2)*(j**2+k**2)/var
                fil[i][j][k]=(1/(2*math.pi*var))*np.exp(p)
    return fil
    



gaussian_blur=make_gaussian_blur(4,(3,5,5))
model=Convolution_Layer(image.shape,gaussian_blur.shape,1)
model.set_weights(gaussian_blur.reshape(1,*gaussian_blur.shape))
out=model.forward(image)
plt.imshow(out[0])



do_nothing=(np.zeros((3,3,3)))
do_nothing[:][1][1]=1
model=Convolution_Layer(image.shape,do_nothing.shape,1)
model.set_weights(do_nothing.reshape(1,*do_nothing.shape))
out=model.forward(image)
print(out.shape)
plt.imshow(out[0])



for i in range(5):
    image = data_batch_1[b'data'][i].reshape(3,32,32).transpose(1,2,0)
    plt.imshow(image)
    plt.show()
    image=image.transpose(2,0,1)
    
    gaussian_blur=make_gaussian_blur(4,(3,5,5))
    model=Convolution_Layer(image.shape,gaussian_blur.shape,1)
    model.set_weights(gaussian_blur.reshape(1,*gaussian_blur.shape))
    out=model.forward(image)
    plt.imshow(out[0])
    plt.show()



C0_weights = np.load("C0_weights.npy")
print(C0_weights.shape)


model=Convolution_Layer((3,32,32),(3,5,5),20)
model.set_weights=C0_weights
C0_out=[]
for i in range(100):
    img=data_batch_1[b'data'][i].reshape(3,32,32)/255
    out=model.forward(img)
    C0_out.append(out)
C0_out=np.array(res)
print(C0_out.shape)



class L2_loss():
    def ___init__(self):
        self.C0_output=None
        self.C_output=None
    
    def forward(self, C0_output,C_output):
        self.C0_output=C0_output
        self.C_output=C_output
        loss=np.sqrt(np.sum((C_output-C0_output)**2))
        
        return loss
    
    def backward(self,output_grad=1):
        
        grad=2*(self.C_output-self.C0_output)
        
        
        return grad


max_epochs=5
model=Convolution_Layer((3,32,32),(3,5,5),20)
L2=L2_loss()
learning_rate=0.000001
temp=[]
for epoch in range(max_epochs):
    
    for i in range(100):
        img=data_batch_1[b'data'][i].reshape(3,32,32)/255
        out=model.forward(img)
        loss=L2.forward(C0_out[i],out)
        print(loss/1000)
        temp.append(loss)
        
        out_grad=L2.backward(loss)
        model.backward(out_grad,learning_rate)
    
    print(f"epoch: {epoch}")

plt.plot(temp)
    

C_out_final=[]
for i in range(100):
    img=data_batch_1[b'data'][i].reshape(3,32,32)/255
    out=model.forward(img)
    C_out_final.append(out)
C_out_final=np.array(res)
print(C_out_final.shape)



