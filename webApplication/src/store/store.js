import { defineStore } from 'pinia';
import { mande } from 'mande'

const museumItems = mande('http://127.0.0.1:8000/items/')
const uploadFile = mande()

export const useMuseumStore = defineStore('museum', {
  state: () => ({
    items: [

    ],
  }),
  getters: {
    doubleCount: (state) => state.count * 2,
  },
  actions: {
    async getItemInfo(itemId){
      const item = await museumItems.get(itemId);
      return item;
    },
    async getSimilarImages(file){
      const formData = new FormData();
      formData.append("file", file, file.name);
      console.log(formData);
      const imageIds = await fetch("http://127.0.0.1:8000/upload_file", {
        method: "POST",
        body: formData,
      })
      let response = (await imageIds.json())['similar_images'];
      console.log(response);
      console.log(10)
      for(let object of response){
        let itemId = Object.keys(object)[0];
        const museumItem = await fetch(`http://127.0.0.1:8000/items/${itemId}`, {
          headers: {"Access-Control-Allow-Origin":"*"}
        })
        const item = await museumItem.json();
        console.log(item);
        // this.items.push()
      }
    }
  },
});