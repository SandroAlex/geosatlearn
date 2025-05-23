// Caminho da pasta no GEE (exemplo: 'users/seu_usuario/sua_pasta')
var folderPath = "projects/hblhug67j2mi/assets/bayer-api-load-tests/";

// Função para deletar todos os assets em uma pasta.
var deleteAssetsInFolder = function(folder) {
  
  var assetsList = ee.data.listAssets(folder);
  if (assetsList.assets && assetsList.assets.length > 0) {
    
    assetsList.assets.forEach(function(asset) {
      print("Deleting asset:", asset.id);
      ee.data.deleteAsset(asset.id);
    });
    
    print("All assets deleted.");
  
    
  } else {
    print("No assets found in folder:", folder);
  }

  
};

// Chame a função passando o caminho da sua pasta.
deleteAssetsInFolder(folderPath);