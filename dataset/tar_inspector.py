import tarfile

tar_path = "/cluster/work/lawecon_repo/gravestones/shards/images/transfer_2025-05-18_084428/transfer_2025-05-18_084428_image_shard-000000.tar"

with tarfile.open(tar_path, "r") as tar:
    members = tar.getmembers()
    print(f"Number of members in the tar file: {len(members)}")
    # get number of members by extension
    jpg_count = sum(1 for member in members if member.name.endswith(".jpg"))
    png_count = sum(1 for member in members if member.name.endswith(".png"))
    json_count = sum(1 for member in members if member.name.endswith(".json"))
    print(f"Number of JPG files: {jpg_count}")
    print(f"Number of PNG files: {png_count}")
    print(f"Number of JSON files: {json_count}")
    # for member in tar.getmembers():
    #     print(member.name)
