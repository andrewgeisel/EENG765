#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Box:
    '''
    This class is a helper class for running a full Bayesian
    filter in 2d.  When you have a PDF, you can store the
    outside limits of the box in this class.  It also has 
    two helper functions that generate a new box from the
    convolution of two boxes (propagate step), and generates a "union" box
    for when you do a multiplication of two boxes (update step)
    '''
    #TODO: Should really make dx a part of this class
    # and then make sure that any box operations (e.g. mult)
    # have the same dx for the two boxes...
    def __init__ (self, x_low=0,x_high=0,y_low=0,y_high=0, dx=None):
        '''
        If dx is None, then x_low... is simply stored in limits
        If not, then x_low ... are "rounded" to the closest 
        dx divisible value.  This is needed if conv and mult
        are going to be used a bit later on
        '''
        assert x_high >= x_low, "Invalid limits for Box"
        assert y_high >= y_low, "Invalid limits for Box"
        self.limits = np.zeros((2,2))
        if dx is not None:
            x_low = np.rint(x_low/dx)*dx
            x_high = np.rint(x_high/dx)*dx
            y_low = np.rint(y_low/dx)*dx
            y_high = np.rint(y_high/dx)*dx
        self.limits[0,0]=x_low
        self.limits[0,1]=x_high
        self.limits[1,0]=y_low
        self.limits[1,1]=y_high

    def gen_xs(self,dx):
        '''
        Takes the limits stored in this class and generates an 
        m X n X 2 array with all the points in the box.  
        m will be determined by dx and the size of the y limits
        n will be determined by dx and the size of the x limits

        Args:
            dx: A single float denoting the size of each "box"
                to be generated
        
        Returns:
            A 3d numpy array, where 2 is the last dimension
        '''
        x,y = self.gen_axes(dx)
        return np.array(np.meshgrid(x, y, indexing='ij')).T
    
    def gen_axes(self,dx):
        '''
        Takes the limits stored in this class and generates 2
        arrays corresponding to the x and y values for this Box.

        Args:
            dx: A single float denoting the size of each "box"
                to be generated
        
        Returns:
            A tuple with of two 1-D numpy arrays, (x,y)
        '''

        x = np.arange(self.limits[0,0],self.limits[0,1]+dx/10.,dx)
        y = np.arange(self.limits[1,0],self.limits[1,1]+dx/10.,dx)
        return (x,y)
    
    def conv(self, box2):
        '''
        This function takes in two boxes and returns a box
        with the correct limits if it is run through a convolution
        with the 'full' option.  This means the output will
        be bigger than either of the inputs

        Args:
            box2: input of type "Box"

        Returns:
            A Box class
        '''
        #Find the middle of the new box
        mid1 = np.sum(self.limits,axis=1)/2.
        mid2 = np.sum(box2.limits,axis=1)/2.
        new_mid = mid1 + mid2
        
        #And how big the box will be
        delta1 = self.limits[:,1] - mid1
        delta2 = box2.limits[:,1] - mid2
        new_delta = delta1+delta2

        return Box(new_mid[0] - new_delta[0], new_mid[0] + new_delta[0] ,
                   new_mid[1] - new_delta[1], new_mid[1] + new_delta[1])

    def mult(self, box2, dx=None):
        '''
        This function takes in two boxes and returns the
        "union" between them, i.e. only the places where there 
        both boxes exist (i.e. two non-zero inputs with a multiply)

        Args:
            self, box2:  Two objects of type "Box"

        Returns:
            a Box class 
        '''
        ### Put your code here (and remove the "pass")
        self_x_low = self.limits[0,0]
        self_x_high = self.limits[0,1]
        self_y_low = self.limits[1,0]
        self_y_high = self.limits[1,1]
        
        box2_x_low = box2.limits[0,0]
        box2_x_high = box2.limits[0,1]
        box2_y_low = box2.limits[1,0]
        box2_y_high = box2.limits[1,1]
        
        new_x_low = max(self_x_low,box2_x_low)
        new_x_high = min(self_x_high,box2_x_high)
        new_y_low = max(self_y_low,box2_y_low)
        new_y_high = min(self_y_high,box2_y_high)

        new_Box = Box(new_x_low,new_x_high,new_y_low,new_y_high,dx)
        return new_Box

    def get_sub_array(self, array, dx, small_box):
        '''
        This function can be used in conjunction with mult.  When
        you have a small box within the current class, and an array
        (with dx) that corresponds with this box, you can pass in
        a smaller box and get back an array for that smaller area.

        Args:
            array:  the bigger array size to get data from.  Should correspond
                in size to the Box class and dx
            dx: the spacing of the array
            small_box:  the smaller box to fit inside this box
        
        Returns:
            an numpy array that is <= the array

        WARNING:  I don't check that array and dx correspond with 
        box.  They should.  Also remember y is the first entry,
        x is the second index.  (So it makes pretty pictures.)
        '''
        assert np.all(small_box.limits[:,0] >= self.limits[:,0]), "small box must fit within this box"
        assert np.all(small_box.limits[:,1] <= self.limits[:,1]), "small box must fit within this box"

        # I do this in two steps as that is the only way I found that I could properly
        #grab sub-blocks out of arrays using numpy.  Maybe there is a better way, but...
        
        #First, grab the right rows... (y values)
        y_low = int(np.rint((small_box.limits[1,0]-self.limits[1,0]) / dx))
        y_high = int(np.rint((small_box.limits[1,1] - self.limits[1,0]) / dx)+1)
        tmp = array[y_low:y_high]

        #Now grab the cols
        x_low = int(np.rint((small_box.limits[0,0]-self.limits[0,0]) / dx))
        x_high = int(np.rint((small_box.limits[0,1]-self.limits[0,0]) / dx) + 1)
        return tmp[:,x_low:x_high]

# Test out box
#%% Test this cell
if __name__ == "__main__":
    #Unfortunately, I can only do grayscale...
    im=cv2.imread("601hge.jpg",cv2.IMREAD_GRAYSCALE)

    #Simple example of what Box does...
    whole_im_box = Box(0,im.shape[1],0,im.shape[0],1)
    subset_box1 = Box(0,400,0,400,1)
    subset_box2 = Box(100,500,0,350,1)
    sub_im1 = whole_im_box.get_sub_array(im,1,subset_box1)
    sub_im2 = whole_im_box.get_sub_array(im,1,subset_box2)
    union_box = subset_box1.mult(subset_box2,1)
    union_im = whole_im_box.get_sub_array(im,1,union_box)
    cv2.imshow('sub image1',sub_im1)
    cv2.imshow('sub image2',sub_im2)
    cv2.imshow('Union image',union_im)
    cv2.waitKey(10000) #Windows will be up for 5 seconds
    cv2.destroyAllWindows()
#%%
    # This code should do the exact same thing as above, but
    # with a different dx
    whole_im_box = Box(0,333.5,0,250,.5)
    subset_box1 = Box(0,200,0,200,.5)
    subset_box2 = Box(50,250,0,175,.5)
    sub_im1 = whole_im_box.get_sub_array(im,.5,subset_box1)
    sub_im2 = whole_im_box.get_sub_array(im,.5,subset_box2)
    union_box = subset_box1.mult(subset_box2,.5)
    union_im = whole_im_box.get_sub_array(im,.5,union_box)
    cv2.imshow('sub image1',sub_im1)
    cv2.imshow('sub image2',sub_im2)
    cv2.imshow('Union image',union_im)
    cv2.imwrite('my_union.png',union_im)
    cv2.waitKey(10000) #Windows will be up for 5 seconds
    cv2.destroyAllWindows()
