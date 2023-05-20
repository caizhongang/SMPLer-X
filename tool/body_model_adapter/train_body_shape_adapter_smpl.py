

def read_deformation_transfer(deformation_transfer_path, device=None, use_normal=False):
    ''' Ref: https://github.com/vchoutas/smplx/blob/bb595ae115c82eee630c9090ad2d9c9691167eee/transfer_model/utils/def_transfer.py#L28-L60
        Reads a deformation transfer
    '''
    if device is None:
        device = torch.device('cpu')
    assert osp.exists(deformation_transfer_path), (
        'Deformation transfer path does not exist:'
        f' {deformation_transfer_path}')
    print(f'Loading deformation transfer from: {deformation_transfer_path}')
    # Read the deformation transfer matrix
    with open(deformation_transfer_path, 'rb') as f:
        def_transfer_setup = pickle.load(f, encoding='latin1')
    if 'mtx' in def_transfer_setup:
        def_matrix = def_transfer_setup['mtx']
        if hasattr(def_matrix, 'todense'):
            def_matrix = def_matrix.todense()
        def_matrix = np.array(def_matrix, dtype=np.float32)
        if not use_normal:
            num_verts = def_matrix.shape[1] // 2
            def_matrix = def_matrix[:, :num_verts]
    elif 'matrix' in def_transfer_setup:
        def_matrix = def_transfer_setup['matrix']
    else:
        valid_keys = ['mtx', 'matrix']
        raise KeyError(f'Deformation transfer setup must contain {valid_keys}')

    def_matrix = torch.tensor(def_matrix, device=device, dtype=torch.float32)
    return def_matrix


def apply_deformation_transfer(def_matrix, vertices, faces, use_normals=False):
    ''' Ref: https://github.com/vchoutas/smplx/blob/bb595ae115c82eee630c9090ad2d9c9691167eee/transfer_model/utils/def_transfer.py#L63-L75
        Applies the deformation transfer on the given meshes
    '''
    if use_normals:
        raise NotImplementedError
    else:
        def_vertices = torch.einsum('mn,bni->bmi', [def_matrix, vertices])
        return def_vertices


def fit_smpl():
    """
    Train model to map smpl neutral/male/female shape to smplx neutral shape
    Ref:
        https://github.com/vchoutas/smplx/blob/bb595ae115c82eee630c9090ad2d9c9691167eee/transfer_model/__main__.py#L36
        https://github.com/vchoutas/smplx/blob/bb595ae115c82eee630c9090ad2d9c9691167eee/transfer_model/transfer_model.py#L257
    """

    smplx_mask_ids_load_path = '/home/alex/github/OSX/tool/body_model_adapter/model_transfer/smplx_mask_ids.npy'
    deformation_transfer_path = '/home/alex/github/OSX/tool/body_model_adapter/model_transfer/smpl2smplx_deftrafo_setup.pkl'

    mask_ids = np.load(smplx_mask_ids_load_path)
    mask_ids = torch.from_numpy(mask_ids).to(device=device)
    def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)

    for gender in ('neutral', 'female', 'male'):
        source = ('smpl', gender)
        target = ('smplx', 'neutral')

        train_dataset = BetasDataset(num_samples=num_samples, range=6)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the neural network and optimizer
        net = BetasAdapter().to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.001)
        # ref: https://github.com/vchoutas/smplx/blob/main/config_files/smpl2smplx.yaml#L11-L14

        # Initialize the body models
        body_model_source = create_body_model(source).to(device)
        print('source model:', body_model_source)
        body_model_target = create_body_model(target).to(device)
        print('target model:', body_model_target)

        # Train the neural network
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, inputs in enumerate(train_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = 10 * net(inputs)

                # compute smpl vertices
                input_vertices = compute_smpl_vertices(body_model_source, betas=inputs)
                input_vertices = apply_deformation_transfer(def_matrix, input_vertices, None, use_normals=False)

                # compute smplx vertices
                output_vertices = compute_smplx_vertices(body_model_target, betas=outputs)

                # mask
                input_vertices = input_vertices[:, mask_ids]
                output_vertices = output_vertices[:, mask_ids]

                loss = per_vertex_loss(input_vertices, output_vertices)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # import pdb; pdb.set_trace()
            # pcd_source = o3d.geometry.PointCloud()
            # pcd_source.points = o3d.utility.Vector3dVector(input_vertices[0].cpu().detach().numpy())
            # pcd_source.paint_uniform_color([0, 1, 0])
            # pcd_target = o3d.geometry.PointCloud()
            # pcd_target.points = o3d.utility.Vector3dVector(output_vertices[0].cpu().detach().numpy())
            # pcd_target.paint_uniform_color([1, 0, 0])
            #
            # # draw correspondences
            # qs, idx, color = [], [], []
            # for pred, gt in zip(pcd_source.points, pcd_target.points):
            #     qs.append(pred)
            #     qs.append(gt)
            #     idx.append([len(qs) - 2, len(qs) - 1])
            #     color.append([0.0, 0, 1.0])
            # lineset = o3d.geometry.LineSet(
            #     points=o3d.utility.Vector3dVector(qs),
            #     lines=o3d.utility.Vector2iVector(idx))
            # lineset.colors = o3d.utility.Vector3dVector(color)
            #
            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([pcd_source, pcd_target, lineset])

            print('Epoch %d loss: %.8f' % (epoch + 1, running_loss / len(train_loader)))

        # save
        save_path = f'{source[0]}_{source[1]}_to_{target[0]}_{target[1]}.pth'
        torch.save(net.state_dict(), save_path)
        print(save_path, 'saved.')