<script>
import {defineComponent} from 'vue'
import LoadingBlinker from "@/components/LoadingBlinker/LoadingBlinker.vue";

export default defineComponent({
    name: "LoadingResults",
    components: {LoadingBlinker},
})
</script>

<template>
    <div class="loading">
        <LoadingBlinker/>
        <div class="warning">
            Подождите,<br>
            идет поиск
        </div>
    </div>
</template>

<style scoped>
.loading {
    margin-top: auto;
    margin-bottom: auto;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

</style>