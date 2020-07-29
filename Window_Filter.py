import sys
import numpy as np

##============================================================================

def window(array_full, size=10, mode='eig', pad=1, stride='default'):
    
    array_full = np.array(array_full)
    comp = lambda x: ((x+size)//size)*size - x

    assert size >= 1
    assert len(array_full.shape) >= 2
    assert pad in [0, 1]
    assert stride == 'default' or type(stride) == int

    left = comp(array_full.shape[1]) // 2
    right = comp(array_full.shape[1]) - left
    top = comp(array_full.shape[0]) // 2
    bottom = comp(array_full.shape[0]) - top  

    output_full = []

##============================================================================
    if len(array_full.shape) == 3:
        
        for i in np.arange(array_full.shape[2]):
            array = array_full[:,:,i]  
            
            if pad == 1:
                array = np.c_[
                    np.ones((array.shape[0], left)), 
                    array,
                    ]
                array = np.c_[
                    array,
                    np.ones((array.shape[0], right)),
                    ]
                array = np.vstack((
                    np.ones((top, array.shape[1])),
                    array,
                    ))    
                array = np.vstack((
                    array,
                    np.ones((bottom, array.shape[1])),
                    ))    
                
            if pad == 0:
                array = np.c_[
                    np.zeros((array.shape[0], left)), 
                    array,
                    ]
                array = np.c_[
                    array,
                    np.zeros((array.shape[0], right)),
                    ]
                array = np.vstack((
                    np.zeros((top, array.shape[1])),
                    array,
                    ))    
                array = np.vstack((
                    array,
                    np.zeros((bottom, array.shape[1])),
                    ))
                
            output = np.array([])
            x1, x2 = 0, 0+size
            y1, y2 = 0, 0+size
            
            if stride == 'default':
                stride = size//2
            
            while True:
                
                win = array[y1:y2, x1:x2]
                if win.shape != (size, size):
                    break
                
                if mode == 'eig':
                    eigen_values = np.linalg.eig(win)[0].real
                    idx = np.argmax(np.abs(eigen_values))
                    decomp = eigen_values[idx]
                    output = np.append(output, [decomp], axis=0)
                else:
                    decomp = mode(win)
                    output = np.append(output, [decomp], axis=0)
                    
                x1, x2 = x1+stride, x2+stride
                if x1 >= array.shape[1] - size+1:
                    x1, x2 = 0, 0+size
                    y1, y2 = y1+stride, y2+stride

            output_full.append(output)  
        output_full = sum(np.array(output_full))
##============================================================================
    elif len(array_full.shape) == 2:
        array = array_full.copy() 
        
        if pad == 1:
            array = np.c_[
                np.ones((array.shape[0], left)), 
                array,
                ]
            array = np.c_[
                array,
                np.ones((array.shape[0], right)),
                ]
            array = np.vstack((
                np.ones((top, array.shape[1])),
                array,
                ))    
            array = np.vstack((
                array,
                np.ones((bottom, array.shape[1])),
                ))    
            
        if pad == 0:
            array = np.c_[
                np.zeros((array.shape[0], left)), 
                array,
                ]
            array = np.c_[
                array,
                np.zeros((array.shape[0], right)),
                ]
            array = np.vstack((
                np.zeros((top, array.shape[1])),
                array,
                ))    
            array = np.vstack((
                array,
                np.zeros((bottom, array.shape[1])),
                ))
            
        x1, x2 = 0, 0+size
        y1, y2 = 0, 0+size
        
        if stride == 'default':
            stride = size//2
        
        while True:
            
            win = array[y1:y2, x1:x2]
            if win.shape != (size, size):
                break

            if mode == 'eig':
                eigen_values = np.linalg.eig(win)[0].real
                idx = np.argmax(np.abs(eigen_values))
                decomp = eigen_values[idx]           
                output_full = np.append(output_full, [decomp], axis=0)
            else:
                decomp = mode(win)
                output_full = np.append(output_full, [decomp], axis=0)

            x1, x2 = x1+stride, x2+stride
            if x1 >= array.shape[1] - size+1:
                x1, x2 = 0, 0+size
                y1, y2 = y1+stride, y2+stride   
##============================================================================

    return np.array(output_full)

##============================================================================
    
