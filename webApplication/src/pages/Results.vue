<script>
import {defineComponent} from 'vue'
import ExhibitCard from "@/components/ExhibitCard/ExhibitCard.vue";
import {useMuseumStore} from "@/store/store.js";

export default defineComponent({
    name: "Results",
    methods: {useMuseumStore},
    components: {ExhibitCard}
})
</script>

<template>
    <div class="results">
        <div class="results__cards">
            <ExhibitCard :id="item.object_id" :key="item.object_id" v-for="item in useMuseumStore().items"/>
        </div>
    </div>
</template>

<style scoped>
.results__cards {
    margin-top: 2.125em;
    margin-bottom: 4.125em;
    display: flex;
    flex-direction: column;
    align-items: center;
    row-gap: 1.25em;
}
</style>