from torch import optim
from tqdm.auto import tqdm
from .helper import *
from .generator import SkipEncoderDecoder, input_noise

def deep_prior(image_path, mask_path, max_dim, reg_noise, input_depth, lr, show_step, training_steps, tqdm_length=100):
    DTYPE = torch.cuda.FloatTensor
    image_np, mask_np = preprocess_images(image_path, mask_path, max_dim)

    print('Building the model ...')
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down=[128]*5,
        num_channels_up=[128]*5,
        num_channels_skip=[128]*5
    ).type(DTYPE)

    objective = torch.nn.MSELoss().type(DTYPE)
    optimizer = optim.Adam(generator.parameters(), lr)

    image_var = np_to_torch_array(image_np).type(DTYPE)
    mask_var = np_to_torch_array(1-mask_np).type(DTYPE)

    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE)

    noise = generator_input.detach().clone()
    generator_input_save = generator_input.detach().clone()

    print('\nStarting training...\n')
    progress_bar = tqdm(range(training_steps), desc = 'Completed', ncols = tqdm_length)

    for step in progress_bar:
        optimizer.zero_grad()
        generator_input = generator_input_save

        if reg_noise>0:
            generator_input = generator_input_save + (noise.normal_() * reg_noise)

        output = generator(generator_input)

        loss = objective(output * mask_var, image_var * mask_var)
        loss.backward()
        if step % show_step == 0:
            output_image= torch_to_np_array(output)
            visualize_sample(image_np, output_image, nrow=2, size_factor=10)
            current_index = len(glob(r'results\*.png'))
            # print(grid.transpose(1,2,0).shape)
            save_img =  (output_image.transpose(1, 2, 0) * 255.0).astype('uint8')
            # print(save_img.shape)
            cv2.imwrite('results\\'+str(current_index)+'.png', save_img)
        
        progress_bar.set_postfix(Loss = loss.item())
        optimizer.step()
    output_image = torch_to_np_array(output)
    visualize_sample(output_image, nrow = 1, size_factor=10)

    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))

    output_path = image_path.split('/')[-1].split('.')[-2] + '-output.jpg'
    print(f'\nSaving final output image to: "{output_path}"\n')

    pil_image.save(output_path)