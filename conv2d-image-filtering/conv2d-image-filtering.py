def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """

    h_image = len(image)
    w_image = len(image[0])

    h_kernel = len(kernel)
    w_kernel = len(kernel[0])
    
    h_out = int((h_image + 2 * padding - h_kernel) / stride + 1)
    w_out = int((w_image + 2 * padding - w_kernel) / stride + 1)

    out = []
    
    for i in range(h_out):
        cal = []
        for j in range(w_out):

            v = 0
            for k in range(h_kernel):
                for l in range(w_kernel):

                    curr_i = i * stride + k - padding
                    curr_j = j * stride + l - padding
                    
                    if curr_i < 0 or curr_i >= h_image or curr_j < 0 or curr_j >= w_image:
                        continue

                    v += (image[curr_i][curr_j] * kernel[k][l])

            cal.append(v)
            

        out.append(cal)

    return out
                    