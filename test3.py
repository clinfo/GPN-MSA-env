import zarr

msa_path = "./89.zarr"
f = zarr.open(msa_path, mode="r")
print(f.keys())
chroms=f.keys()
data = {chrom: f[chrom] for chrom in chroms}
print(type(data["1"]))
print(data["1"].shape)


