<script>
import {defineComponent} from 'vue'
import CommonIcon from "@/components/Common/CommonIcon.vue";
import {useMuseumStore} from "@/store/store.js";

export default defineComponent({
    name: "PhotoUploader1",
    components: {CommonIcon},
    data(){
        return {
            store: useMuseumStore(),
        }
    },
    methods: {
        uploadFile(){
            this.selectFile()
        },
        async handleFile(e){
            let file = e.target.files[0];
            if(file){
                const imageIds = await this.store.getSimilarImages(file);
                this.$router.push('/results');
                this.$emit('loading-file')
                console.log(imageIds);
            }
            console.log(file);
            this.$refs.fileInput.removeEventListener("change", this.handleFile);
        },
        selectFile(){
            const fileInput = this.$refs.fileInput;
            fileInput.click();
            fileInput.addEventListener("change", this.handleFile)
        }
    }
})
</script>

<template>
    <div @click="uploadFile" class="photo_uploader">
        <div class="photo_uploader__inner">
            <CommonIcon class="photo_uploader__icon" icon-file="camera.svg"/>
            <div class="photo_uploader__text">
                Добавить фото
            </div>
            <input ref="fileInput" class="file_input" type="file" >
        </div>
    </div>
</template>

<style scoped>
.file_input {
    display: none;
}
.photo_uploader {
    cursor: pointer;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background-color: var(--fifth-color);
    width: 17.8125em;
    height: 18.75em;
    border-radius: 0.625em;
}
.photo_uploader__inner {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.photo_uploader__text {
    font-size: var(--secondary-font-size);
    font-family: var(--secondary-font-family);
}
.photo_uploader__icon {
    height: 2.5em;
}
</style>