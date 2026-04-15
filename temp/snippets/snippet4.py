samples = [train_dataset[0], test_dataset[3000]]

for sample in samples:
    print(sample)
    print('Sample check:')
    print(f"  image shape: {tuple(sample['image'].shape)}")

    obj_class = sample['object_class'].item()
    motion_class = sample['motion_class'].item()
    force_n = float(sample['force_n'])

    # Get string from label mapping
    object_name = idx2label['object'][obj_class]
    motion_name = idx2label['motion'][motion_class]
    print(f"  object name: {object_name}")
    print(f"  motion name: {motion_name}")
    print(f"  force  {force_n:.4f}")

    materials_vector = sample['materials_vector']
    print(f"  materials multi-hot: {materials_vector}")

    # works for list/tuple/tensor
    active_materials = utils.decode_materials(materials_vector, materials_list)
    print(active_materials)  # e.g. ['b', 'c']

# Print image tensor stats to verify background subtraction worked
    img_tensor = sample['image']
    print(
        f"  image tensor stats: min={img_tensor.min():.4f}, max={img_tensor.max():.4f}, mean={img_tensor.mean():.4f}")

# loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
# batch = next(iter(loader))
# print('Batch check:')
# print(f"  image batch shape: {tuple(batch['image'].shape)}")
# print(f"  object batch shape: {tuple(batch['object_class'].shape)}")
# print(f"  force batch shape: {tuple(batch['force'].shape)}")
