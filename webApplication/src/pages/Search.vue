<script>
import {defineComponent} from 'vue'
import PhotoUploader from "@/components/PhotoUploader/PhotoUploader.vue";
import LoadingResults from "@/components/LoadingBlinker/LoadingResults.vue";

export default defineComponent({
    name: "Search",
    components: {LoadingResults, PhotoUploader},
    data(){
        return {
            isLoading: false,
        }
    }
})
</script>

<template>
    <div class="search_page">
        <div class="search_page__header">
            <span class="header">Поиск</span>
        </div>
        <div class="search_page__files" v-show="!isLoading">
            <PhotoUploader @loading-file="isLoading = true" class="search_page__uploader"/>
            <div class="search_page__warning">
                <span class="warning">*Добавьте фото, чтобы получить список схожих экспонатов</span>
            </div>
        </div>
        <LoadingResults class="blinker" v-show="isLoading"/>
    </div>
</template>

<style scoped>
.search_page {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.blinker {
    margin-top: 10em;
}
.search_page__files {

}
.header {
    font-size: var(--primary-font-size);
    font-family: var(--primary-font-family), serif;
    color: var(--second-color);
    letter-spacing: 1.4px;
}
.search_page__header {
    margin-top: 1.25em;
    margin-bottom: 4.0625em;
}
.search_page__uploader {

}
.search_page__warning {
    margin-top: 1.875em;
    margin-bottom: 4.0625em;
    max-width: 17.8125em;
}

</style>